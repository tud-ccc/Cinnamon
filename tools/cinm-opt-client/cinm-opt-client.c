/// Drop-in replacement for cinm-opt that runs each invocation in a cinm-opt
/// fork server (see tools/cinm-opt/OptServer.h), which skips cinm-opt's
/// startup cost.
///
/// If no server is running, this starts one in the background and runs
/// cinm-opt directly for the current invocation. Any failure to reach the
/// server also falls back to running cinm-opt directly, so callers see the
/// same behaviour either way.
///
/// Environment:
///   CINM_OPT_BINARY     cinm-opt to use; default: next to this executable.
///   CINM_OPT_SOCKET     server socket; default: one per cinm-opt binary in
///                       $XDG_RUNTIME_DIR, or else in /tmp/cinm-opt-<uid>.
///   CINM_OPT_NO_SERVER  if non-empty, always run cinm-opt directly.
///   CINM_OPT_SERVER_IDLE_SECONDS
///                       read by the server: exit after this long without
///                       requests (default 900, 0 means never).
///
/// @file
/// @author      Clément Fournier (clement.fournier@tu-dresden.de)

#define _GNU_SOURCE

#include "OptServerProtocol.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

static char optPath[PATH_MAX];
static int serverSock = -1;

_Noreturn static void runDirectly(char **argv) {
  execv(optPath, argv);
  fprintf(stderr, "cinm-opt-client: cannot execute %s: %s\n", optPath,
          strerror(errno));
  exit(127);
}

/// Resolves the cinm-opt binary into `optPath`.
static int findCinmOpt(void) {
  const char *override = getenv("CINM_OPT_BINARY");
  if (override && *override) {
    if (!realpath(override, optPath))
      return -1;
    return 0;
  }
  char self[PATH_MAX];
  ssize_t n = readlink("/proc/self/exe", self, sizeof self - 1);
  if (n < 0)
    return -1;
  self[n] = '\0';
  char *slash = strrchr(self, '/');
  if (!slash)
    return -1;
  *slash = '\0';
  if (snprintf(optPath, sizeof optPath, "%s/cinm-opt", self) >=
      (int)sizeof optPath)
    return -1;
  return 0;
}

/// Checks that `dir` is a directory only this user can access.
static int isPrivateDir(const char *dir) {
  struct stat st;
  return lstat(dir, &st) == 0 && S_ISDIR(st.st_mode) && st.st_uid == getuid() &&
         (st.st_mode & 077) == 0;
}

/// Picks the socket path: one per cinm-opt binary, so that servers for
/// different build directories don't mix.
static int socketPathFor(char *out, size_t size) {
  const char *override = getenv("CINM_OPT_SOCKET");
  if (override && *override) {
    if (snprintf(out, size, "%s", override) < (int)size)
      return 0;
    fprintf(stderr,
            "cinm-opt-client: CINM_OPT_SOCKET is longer than %zu bytes, "
            "running cinm-opt directly\n",
            size - 1);
    return -1;
  }

  char dir[PATH_MAX];
  const char *runtime = getenv("XDG_RUNTIME_DIR");
  if (runtime && *runtime && isPrivateDir(runtime)) {
    snprintf(dir, sizeof dir, "%s", runtime);
  } else {
    snprintf(dir, sizeof dir, "/tmp/cinm-opt-%u", (unsigned)getuid());
    if (mkdir(dir, 0700) != 0 && errno != EEXIST)
      return -1;
    if (!isPrivateDir(dir))
      return -1;
  }

  // FNV-1a
  uint64_t hash = 0xcbf29ce484222325ull;
  for (const char *p = optPath; *p; ++p)
    hash = (hash ^ (unsigned char)*p) * 0x100000001b3ull;
  return snprintf(out, size, "%s/cinm-opt-%016llx.sock", dir,
                  (unsigned long long)hash) < (int)size
             ? 0
             : -1;
}

/// Starts a detached server on `socketPath`, logging to `<socketPath>.log`.
/// If several clients race to do this, all but one server exit immediately.
static void startServer(const char *socketPath) {
  pid_t child = fork();
  if (child < 0)
    return;
  if (child > 0) {
    waitpid(child, NULL, 0);
    return;
  }

  setsid();
  if (fork() != 0)
    _exit(0);

  char logPath[PATH_MAX];
  snprintf(logPath, sizeof logPath, "%s.log", socketPath);
  int devNull = open("/dev/null", O_RDONLY);
  int log = open(logPath, O_WRONLY | O_CREAT | O_APPEND, 0600);
  if (devNull < 0 || log < 0)
    _exit(1);
  dup2(devNull, 0);
  dup2(log, 1);
  dup2(log, 2);
  for (int fd = 3; fd < 1024; ++fd)
    close(fd);
  execl(optPath, optPath, "--serve", socketPath, (char *)NULL);
  _exit(127);
}

static int connectTo(const char *socketPath) {
  struct sockaddr_un addr;
  memset(&addr, 0, sizeof addr);
  addr.sun_family = AF_UNIX;
  if (strlen(socketPath) >= sizeof addr.sun_path) {
    errno = ENAMETOOLONG;
    return -1;
  }
  strcpy(addr.sun_path, socketPath);
  int sock = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (sock < 0)
    return -1;
  if (connect(sock, (struct sockaddr *)&addr, sizeof addr) != 0) {
    int err = errno;
    close(sock);
    errno = err;
    return -1;
  }
  return sock;
}

/// Appends `prefix` and the NUL-terminated `s` to the payload buffer.
static int append(char **buf, size_t *len, size_t *cap, const char *prefix,
                  const char *s) {
  size_t prefixLen = strlen(prefix);
  size_t need = prefixLen + strlen(s) + 1;
  if (*len + need > *cap) {
    size_t newCap = (*cap ? *cap * 2 : 4096) + need;
    char *grown = realloc(*buf, newCap);
    if (!grown)
      return -1;
    *buf = grown;
    *cap = newCap;
  }
  memcpy(*buf + *len, prefix, prefixLen);
  memcpy(*buf + *len + prefixLen, s, need - prefixLen);
  *len += need;
  return 0;
}

static int sendFull(int sock, const void *buf, size_t size) {
  const char *p = buf;
  while (size > 0) {
    ssize_t n = send(sock, p, size, MSG_NOSIGNAL);
    if (n < 0 && errno == EINTR)
      continue;
    if (n <= 0)
      return -1;
    p += n;
    size -= n;
  }
  return 0;
}

static int sendRequest(int sock, char **argv) {
  char *payload = NULL;
  size_t len = 0, cap = 0;
  char cwd[PATH_MAX];
  if (!getcwd(cwd, sizeof cwd) || append(&payload, &len, &cap, "", cwd) != 0)
    return -1;
  for (char **a = argv; *a; ++a)
    if (append(&payload, &len, &cap, "A", *a) != 0)
      return -1;
  for (char **e = environ; *e; ++e)
    if (append(&payload, &len, &cap, "E", *e) != 0)
      return -1;
  if (len > CINM_OPT_SERVER_MAX_PAYLOAD)
    return -1;

  mode_t mask = umask(0);
  umask(mask);
  struct cinm_opt_request_header header = {CINM_OPT_SERVER_MAGIC,
                                           CINM_OPT_SERVER_VERSION,
                                           (uint32_t)mask, (uint32_t)len};

  // Send stdin, stdout and stderr with the first byte of the header.
  int fds[3] = {0, 1, 2};
  char control[CMSG_SPACE(sizeof fds)];
  memset(control, 0, sizeof control);
  struct iovec iov = {&header, 1};
  struct msghdr msg = {0};
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_control = control;
  msg.msg_controllen = sizeof control;
  struct cmsghdr *c = CMSG_FIRSTHDR(&msg);
  c->cmsg_level = SOL_SOCKET;
  c->cmsg_type = SCM_RIGHTS;
  c->cmsg_len = CMSG_LEN(sizeof fds);
  memcpy(CMSG_DATA(c), fds, sizeof fds);
  ssize_t n;
  do
    n = sendmsg(sock, &msg, MSG_NOSIGNAL);
  while (n < 0 && errno == EINTR);
  if (n != 1)
    return -1;

  int rc = sendFull(sock, (char *)&header + 1, sizeof header - 1);
  if (rc == 0)
    rc = sendFull(sock, payload, len);
  free(payload);
  return rc;
}

/// Reads one reply. Returns 1 on success, 0 on end of stream, -1 on error.
static int readReply(int sock, struct cinm_opt_reply *reply) {
  char *p = (char *)reply;
  size_t got = 0;
  while (got < sizeof *reply) {
    ssize_t n = read(sock, p + got, sizeof *reply - got);
    if (n < 0 && errno == EINTR)
      continue;
    if (n < 0)
      return -1;
    if (n == 0)
      return got == 0 ? 0 : -1;
    got += n;
  }
  return 1;
}

/// Passes signals meant for cinm-opt on to the worker.
static void forwardSignal(int sig) {
  unsigned char byte = (unsigned char)sig;
  (void)!send(serverSock, &byte, 1, MSG_NOSIGNAL | MSG_DONTWAIT);
}

_Noreturn static void dieBySignal(int sig) {
  struct rlimit noCore = {0, 0};
  setrlimit(RLIMIT_CORE, &noCore);
  signal(sig, SIG_DFL);
  sigset_t set;
  sigemptyset(&set);
  sigaddset(&set, sig);
  sigprocmask(SIG_UNBLOCK, &set, NULL);
  raise(sig);
  exit(128 + sig);
}

int main(int argc, char **argv) {
  (void)argc;
  if (findCinmOpt() != 0) {
    fprintf(stderr, "cinm-opt-client: cannot locate cinm-opt\n");
    return 127;
  }
  const char *noServer = getenv("CINM_OPT_NO_SERVER");
  if (noServer && *noServer)
    runDirectly(argv);

  char socketPath[sizeof(((struct sockaddr_un *)0)->sun_path)];
  if (socketPathFor(socketPath, sizeof socketPath) != 0)
    runDirectly(argv);

  serverSock = connectTo(socketPath);
  if (serverSock < 0) {
    if (errno == ENOENT || errno == ECONNREFUSED)
      startServer(socketPath);
    runDirectly(argv);
  }

  struct sigaction forward;
  memset(&forward, 0, sizeof forward);
  forward.sa_handler = forwardSignal;
  forward.sa_flags = SA_RESTART;
  sigaction(SIGINT, &forward, NULL);
  sigaction(SIGTERM, &forward, NULL);
  sigaction(SIGHUP, &forward, NULL);
  sigaction(SIGQUIT, &forward, NULL);

  struct cinm_opt_reply reply;
  if (sendRequest(serverSock, argv) != 0 ||
      readReply(serverSock, &reply) != 1 ||
      reply.kind != CINM_OPT_REPLY_STARTED) {
    // Nothing ran yet, so running cinm-opt ourselves is safe.
    close(serverSock);
    runDirectly(argv);
  }

  if (readReply(serverSock, &reply) != 1) {
    fprintf(stderr, "cinm-opt-client: lost the connection to the server\n");
    return 125;
  }
  if (reply.kind == CINM_OPT_REPLY_SIGNALED)
    dieBySignal(reply.value);
  return reply.value;
}
