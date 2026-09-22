/// Fork server mode of cinm-opt. See OptServer.h and OptServerProtocol.h.
///
/// Process structure: the server accepts a connection and forks a handler,
/// which reads the request and forks the worker. The handler owns the
/// connection for the worker's lifetime: it forwards signals, kills the
/// worker if the client goes away, and reports how the worker terminated.
/// The server itself never waits on anything but its listening socket, and
/// its handlers survive when it re-executes itself.
///
/// The server must stay single-threaded, since fork() only copies the calling
/// thread. MLIRContext creates its thread pool lazily, in the worker.
///
/// @file
/// @author      Clément Fournier (clement.fournier@tu-dresden.de)

#include "OptServer.h"
#include "OptServerProtocol.h"

#include <cerrno>
#include <climits>
#include <csignal>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

// Missing from older kernel headers; the number is the same on every
// architecture.
#ifndef SYS_pidfd_open
#define SYS_pidfd_open 434
#endif

using namespace cinm_opt;

namespace {

constexpr int kDefaultIdleSeconds =
    1 * 60; // Shutdown server if idle for 1 minute

void logf(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
void logf(const char *fmt, ...) {
  char stamp[32];
  time_t now = time(nullptr);
  struct tm tm;
  strftime(stamp, sizeof stamp, "%F %T", localtime_r(&now, &tm));
  fprintf(stderr, "[%s cinm-opt-server %d] ", stamp, (int)getpid());
  va_list args;
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  va_end(args);
  fputc('\n', stderr);
}

bool readFull(int fd, void *buf, size_t size) {
  auto *p = static_cast<char *>(buf);
  while (size > 0) {
    ssize_t n = read(fd, p, size);
    if (n < 0 && errno == EINTR)
      continue;
    if (n <= 0)
      return false;
    p += n;
    size -= n;
  }
  return true;
}

bool writeFull(int fd, const void *buf, size_t size) {
  const auto *p = static_cast<const char *>(buf);
  while (size > 0) {
    ssize_t n = write(fd, p, size);
    if (n < 0 && errno == EINTR)
      continue;
    if (n <= 0)
      return false;
    p += n;
    size -= n;
  }
  return true;
}

bool sendReply(int sock, cinm_opt_reply_kind kind, int value) {
  cinm_opt_reply reply{static_cast<uint32_t>(kind), value};
  return writeFull(sock, &reply, sizeof reply);
}

//===----------------------------------------------------------------------===//
// Detecting rebuilds
//===----------------------------------------------------------------------===//

/// A file mapped into this process, and its identity when it was loaded.
struct MappedFile {
  std::string path;
  ino_t inode;
  off_t size;
  struct timespec mtime;
};

/// Lists the files mapped into this process: the executable, every shared
/// library, and the dynamic loader.
///
/// The inode comes from /proc/self/maps, so a file that was replaced by a new
/// one (which is how linkers write their output) is recognised even if that
/// happened before this runs. Size and mtime come from a stat() here, to also
/// catch files overwritten in place.
std::vector<MappedFile> snapshotMappedFiles() {
  std::vector<MappedFile> files;
  std::set<std::string> seen;
  std::ifstream maps("/proc/self/maps");
  std::string line;
  while (std::getline(maps, line)) {
    // address perms offset dev inode path
    size_t slash = line.find('/');
    if (slash == std::string::npos)
      continue;
    unsigned long inode = 0;
    if (sscanf(line.c_str(), "%*s %*s %*s %*s %lu", &inode) != 1 || inode == 0)
      continue;
    std::string path = line.substr(slash);
    if (!seen.insert(path).second)
      continue;

    MappedFile file{path, static_cast<ino_t>(inode), -1, {0, 0}};
    struct stat st;
    if (stat(path.c_str(), &st) == 0 && st.st_ino == file.inode) {
      file.size = st.st_size;
      file.mtime = st.st_mtim;
    }
    files.push_back(std::move(file));
  }
  return files;
}

/// Returns the first mapped file that no longer matches what was loaded, or
/// null if all of them still match.
const MappedFile *findChangedFile(const std::vector<MappedFile> &files) {
  for (const MappedFile &f : files) {
    struct stat st;
    if (stat(f.path.c_str(), &st) != 0 || st.st_ino != f.inode ||
        st.st_size != f.size || st.st_mtim.tv_sec != f.mtime.tv_sec ||
        st.st_mtim.tv_nsec != f.mtime.tv_nsec)
      return &f;
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Handling one request
//===----------------------------------------------------------------------===//

struct Request {
  cinm_opt_request_header header;
  int stdioFds[3] = {-1, -1, -1};
  /// Backing storage for all the strings below.
  std::vector<char> payload;
  const char *cwd = nullptr;
  std::vector<char *> argv;
  std::vector<char *> env;
};

bool receiveRequest(int sock, Request &req) {
  // The file descriptors travel with the first byte of the header.
  char control[CMSG_SPACE(3 * sizeof(int))];
  struct iovec iov = {&req.header, sizeof req.header};
  struct msghdr msg = {};
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_control = control;
  msg.msg_controllen = sizeof control;
  ssize_t n;
  do
    n = recvmsg(sock, &msg, MSG_CMSG_CLOEXEC);
  while (n < 0 && errno == EINTR);
  if (n <= 0)
    return false;

  for (struct cmsghdr *c = CMSG_FIRSTHDR(&msg); c; c = CMSG_NXTHDR(&msg, c)) {
    if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_RIGHTS &&
        c->cmsg_len == CMSG_LEN(3 * sizeof(int)))
      memcpy(req.stdioFds, CMSG_DATA(c), 3 * sizeof(int));
  }
  if (req.stdioFds[0] < 0 || (msg.msg_flags & MSG_CTRUNC)) {
    logf("request without stdio file descriptors");
    return false;
  }

  if (!readFull(sock, reinterpret_cast<char *>(&req.header) + n,
                sizeof req.header - n))
    return false;
  if (req.header.magic != CINM_OPT_SERVER_MAGIC ||
      req.header.version != CINM_OPT_SERVER_VERSION ||
      req.header.payload_size > CINM_OPT_SERVER_MAX_PAYLOAD) {
    logf("rejecting request with protocol %#x version %u", req.header.magic,
         req.header.version);
    return false;
  }

  req.payload.resize(req.header.payload_size + 1);
  if (!readFull(sock, req.payload.data(), req.header.payload_size))
    return false;
  req.payload.back() = '\0';

  char *p = req.payload.data();
  char *end = p + req.header.payload_size;
  req.cwd = p;
  p += strlen(p) + 1;
  while (p < end) {
    char *s = p;
    p += strlen(p) + 1;
    if (*s == 'A')
      req.argv.push_back(s + 1);
    else if (*s == 'E')
      req.env.push_back(s + 1);
  }
  if (req.argv.empty()) {
    logf("request without arguments");
    return false;
  }
  req.argv.push_back(nullptr);
  return true;
}

/// Turns this process into the worker for `req`. Does not return.
[[noreturn]] void runWorker(int sock, Request &req, RunRequestFn runRequest) {
  close(sock);
  for (int i = 0; i < 3; ++i) {
    dup2(req.stdioFds[i], i);
    close(req.stdioFds[i]);
  }
  signal(SIGPIPE, SIG_DFL);
  signal(SIGCHLD, SIG_DFL);

  if (chdir(req.cwd) != 0) {
    fprintf(stderr, "cinm-opt: cannot enter %s: %s\n", req.cwd,
            strerror(errno));
    _exit(127);
  }
  umask(req.header.umask);
  clearenv();
  for (char *var : req.env)
    putenv(var);

  // Leave through exit() rather than _exit() so that buffered output such as
  // llvm::outs() is flushed.
  exit(runRequest(static_cast<int>(req.argv.size()) - 1, req.argv.data()));
}

/// Handles one connection in a process forked for it. Returns the exit code
/// of that process.
int handleConnection(int sock, RunRequestFn runRequest) {
  signal(SIGCHLD, SIG_DFL);

  Request req;
  if (!receiveRequest(sock, req))
    return 1;

  pid_t worker = fork();
  if (worker < 0) {
    logf("fork: %s", strerror(errno));
    return 1;
  }
  if (worker == 0)
    runWorker(sock, req, runRequest);

  for (int fd : req.stdioFds)
    close(fd);
  if (!sendReply(sock, CINM_OPT_REPLY_STARTED, 0)) {
    kill(worker, SIGKILL);
    waitpid(worker, nullptr, 0);
    return 1;
  }

  int pidfd = static_cast<int>(syscall(SYS_pidfd_open, worker, 0));
  while (pidfd >= 0) {
    struct pollfd fds[2] = {{sock, POLLIN, 0}, {pidfd, POLLIN, 0}};
    if (poll(fds, 2, -1) < 0) {
      if (errno == EINTR)
        continue;
      break;
    }
    if (fds[1].revents)
      break;
    if (fds[0].revents) {
      unsigned char sig;
      ssize_t n = read(sock, &sig, 1);
      if (n < 0 && errno == EINTR)
        continue;
      if (n <= 0) {
        // The client went away: nobody is waiting for the result.
        kill(worker, SIGKILL);
        break;
      }
      kill(worker, sig);
    }
  }
  if (pidfd >= 0)
    close(pidfd);

  int status;
  while (waitpid(worker, &status, 0) < 0) {
    if (errno != EINTR)
      return 1;
  }
  if (WIFSIGNALED(status))
    sendReply(sock, CINM_OPT_REPLY_SIGNALED, WTERMSIG(status));
  else
    sendReply(sock, CINM_OPT_REPLY_EXITED, WEXITSTATUS(status));
  return 0;
}

//===----------------------------------------------------------------------===//
// Server
//===----------------------------------------------------------------------===//

struct ServerFds {
  int lock = -1;
  int listen = -1;
};

/// Takes over the lock file and listening socket from the image that
/// re-executed this one.
bool inheritFds(ServerFds &fds) {
  const char *inherited = getenv(CINM_OPT_SERVER_FDS_ENV);
  if (!inherited)
    return false;
  bool ok = sscanf(inherited, "%d,%d", &fds.lock, &fds.listen) == 2;
  unsetenv(CINM_OPT_SERVER_FDS_ENV);
  if (!ok)
    return false;
  fcntl(fds.lock, F_SETFD, FD_CLOEXEC);
  fcntl(fds.listen, F_SETFD, FD_CLOEXEC);
  return true;
}

enum class OpenResult { Ok, AlreadyServing, Error };

/// Binds a fresh listening socket at `socketPath`. The lock file next to it
/// ensures a single server per socket, which makes it safe to remove a
/// socket left behind by a server that died.
OpenResult openFds(const std::string &socketPath, ServerFds &fds) {
  std::string lockPath = socketPath + ".lock";
  fds.lock = open(lockPath.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
  if (fds.lock < 0) {
    logf("cannot open %s: %s", lockPath.c_str(), strerror(errno));
    return OpenResult::Error;
  }
  if (flock(fds.lock, LOCK_EX | LOCK_NB) != 0) {
    if (errno == EWOULDBLOCK)
      return OpenResult::AlreadyServing;
    logf("cannot lock %s: %s", lockPath.c_str(), strerror(errno));
    return OpenResult::Error;
  }

  struct sockaddr_un addr = {};
  addr.sun_family = AF_UNIX;
  if (socketPath.size() >= sizeof addr.sun_path) {
    logf("socket path too long: %s", socketPath.c_str());
    return OpenResult::Error;
  }
  memcpy(addr.sun_path, socketPath.c_str(), socketPath.size() + 1);
  unlink(socketPath.c_str());
  fds.listen = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (fds.listen < 0 ||
      bind(fds.listen, reinterpret_cast<struct sockaddr *>(&addr),
           sizeof addr) != 0 ||
      listen(fds.listen, SOMAXCONN) != 0) {
    logf("cannot listen on %s: %s", socketPath.c_str(), strerror(errno));
    return OpenResult::Error;
  }
  return OpenResult::Ok;
}

int countThreads() {
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    if (line.rfind("Threads:", 0) == 0)
      return atoi(line.c_str() + 8);
  }
  return -1;
}

/// Replaces this process with a fresh image of the executable, which takes
/// over the socket. Connections already queued on it are kept.
[[noreturn]] void reexec(const std::string &exePath, char **argv,
                         const ServerFds &fds, const std::string &socketPath) {
  fcntl(fds.lock, F_SETFD, 0);
  fcntl(fds.listen, F_SETFD, 0);
  std::string inherited =
      std::to_string(fds.lock) + "," + std::to_string(fds.listen);
  setenv(CINM_OPT_SERVER_FDS_ENV, inherited.c_str(), 1);
  execv(exePath.c_str(), argv);

  logf("cannot re-execute %s: %s", exePath.c_str(), strerror(errno));
  unlink(socketPath.c_str());
  _exit(1);
}

} // namespace

int cinm_opt::runServer(int argc, char **argv, RunRequestFn runRequest) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s --serve <socket path>\n", argv[0]);
    return 2;
  }
  std::string socketPath = argv[2];

  char exe[PATH_MAX];
  ssize_t exeLen = readlink("/proc/self/exe", exe, sizeof exe - 1);
  if (exeLen < 0) {
    logf("cannot resolve /proc/self/exe: %s", strerror(errno));
    return 1;
  }
  std::string exePath(exe, exeLen);

  // OpenBLAS, linked in through mlpack, starts a thread per core from its
  // load-time constructor unless told otherwise. Only the environment
  // variable can prevent that, so set it and start over. Workers then run
  // BLAS single-threaded, whatever the client's environment says.
  if (countThreads() != 1 && !getenv("OPENBLAS_NUM_THREADS")) {
    setenv("OPENBLAS_NUM_THREADS", "1", 1);
    execv(exePath.c_str(), argv);
    logf("cannot re-execute %s: %s", exePath.c_str(), strerror(errno));
    return 1;
  }
  if (int threads = countThreads(); threads != 1) {
    logf("refusing to serve: %d threads before forking", threads);
    return 1;
  }

  int idleSeconds = kDefaultIdleSeconds;
  if (const char *idle = getenv("CINM_OPT_SERVER_IDLE_SECONDS"))
    idleSeconds = atoi(idle);

  ServerFds fds;
  bool reexecuted = inheritFds(fds);
  if (!reexecuted) {
    switch (openFds(socketPath, fds)) {
    case OpenResult::Ok:
      break;
    case OpenResult::AlreadyServing:
      return 0;
    case OpenResult::Error:
      return 1;
    }
  }

  if (chdir("/") != 0)
    logf("cannot enter /: %s", strerror(errno));
  signal(SIGPIPE, SIG_IGN);
  struct sigaction reap = {};
  reap.sa_handler = SIG_IGN;
  reap.sa_flags = SA_NOCLDWAIT;
  sigaction(SIGCHLD, &reap, nullptr);

  std::vector<MappedFile> mapped = snapshotMappedFiles();
  logf("%s on %s, watching %zu mapped files",
       reexecuted ? "reloaded" : "serving", socketPath.c_str(), mapped.size());

  while (true) {
    struct pollfd pfd = {fds.listen, POLLIN, 0};
    int ready = poll(&pfd, 1, idleSeconds > 0 ? idleSeconds * 1000 : -1);
    if (ready < 0) {
      if (errno == EINTR)
        continue;
      logf("poll: %s", strerror(errno));
      unlink(socketPath.c_str());
      return 1;
    }
    if (ready == 0) {
      logf("idle for %d s, exiting", idleSeconds);
      unlink(socketPath.c_str());
      return 0;
    }

    if (const MappedFile *changed = findChangedFile(mapped)) {
      logf("%s changed, reloading", changed->path.c_str());
      reexec(exePath, argv, fds, socketPath);
    }

    int conn = accept4(fds.listen, nullptr, nullptr, SOCK_CLOEXEC);
    if (conn < 0)
      continue;
    pid_t handler = fork();
    if (handler < 0)
      logf("fork: %s", strerror(errno));
    if (handler == 0) {
      close(fds.listen);
      close(fds.lock);
      _exit(handleConnection(conn, runRequest));
    }
    close(conn);
  }
}
