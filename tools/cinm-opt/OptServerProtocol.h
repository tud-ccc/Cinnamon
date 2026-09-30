/// Wire protocol between cinm-opt-client and the cinm-opt fork server.
///
/// This header is plain C: the client is a C program that does not link
/// against LLVM, so that it starts in about a millisecond.
///
/// A request is a single connection on the server's Unix socket:
///
///  1. The client sends a cinm_opt_request_header. Its first byte carries an
///     SCM_RIGHTS message with the client's stdin, stdout and stderr.
///  2. The client sends `payload_size` bytes of NUL-terminated strings: the
///     working directory, then one string per argument prefixed with 'A', and
///     one per environment variable prefixed with 'E'.
///  3. The server replies CINM_OPT_REPLY_STARTED once a worker process owns
///     the request, then CINM_OPT_REPLY_EXITED or CINM_OPT_REPLY_SIGNALED when
///     it terminates.
///  4. While the worker runs, each byte the client sends is a signal number
///     that the server delivers to the worker. Closing the connection kills
///     the worker.
///
/// If the connection closes before CINM_OPT_REPLY_STARTED, no worker ran and
/// the client may run cinm-opt directly instead.
///
/// @file
/// @author      Clément Fournier (clement.fournier@tu-dresden.de)

#ifndef CINM_OPT_SERVER_PROTOCOL_H
#define CINM_OPT_SERVER_PROTOCOL_H

#include <stdint.h>

#define CINM_OPT_SERVER_MAGIC 0x434e4d4fu
#define CINM_OPT_SERVER_VERSION 1u

/// Upper bound on the payload, which holds argv and the environment.
#define CINM_OPT_SERVER_MAX_PAYLOAD (16u << 20)

/// Environment variable through which the server hands its listening socket
/// and lock file to the image it re-executes, as "<lock fd>,<listen fd>".
#define CINM_OPT_SERVER_FDS_ENV "CINM_OPT_SERVER_FDS"

struct cinm_opt_request_header {
  uint32_t magic;
  uint32_t version;
  uint32_t umask;
  uint32_t payload_size;
};

enum cinm_opt_reply_kind {
  CINM_OPT_REPLY_STARTED = 1,
  /// `value` is the exit code.
  CINM_OPT_REPLY_EXITED = 2,
  /// `value` is the signal number.
  CINM_OPT_REPLY_SIGNALED = 3,
};

struct cinm_opt_reply {
  uint32_t kind;
  int32_t value;
};

#endif // CINM_OPT_SERVER_PROTOCOL_H
