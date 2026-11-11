#ifdef UPMEM_RT_STATS

#include "timers.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// ─────────────────────────────────────────────────────────────────────────────
// Record types
// ─────────────────────────────────────────────────────────────────────────────

typedef struct {
  int iteration;
  uint64_t elapsed_ns;
  size_t bytes_per_dpu;
  uint32_t num_dpus;
  size_t num_blocks;
  // Which transfer op produced this row: see upmemrt_record_scatter.
  const char *kind;
  // User-supplied tag from upmem.timing_tag, or "" if none.
  const char *tag;
} XferRecord;

typedef struct {
  int iteration;
  uint64_t elapsed_ns;
  uint32_t num_dpus;
} LaunchRecord;

typedef struct {
  int iteration;
  uint64_t elapsed_ns;
  size_t bytes;
} CopyRecord;

typedef struct {
  int iteration;
  uint64_t elapsed_ns;
  uint32_t num_dpus;
  // Whether a node bracket was open: the load is then part of that node's
  // time (a set timeshared between programs reloads inside its nodes),
  // which a reader must not add again.
  int in_node;
} LoadRecord;

typedef struct {
  int iteration;
  uint64_t elapsed_ns;
  size_t bytes;
  const char *kind;
} CompactRecord;

typedef struct {
  int iteration;
  uint64_t elapsed_ns;
  const char *function;
} HostRecord;

typedef struct {
  int iteration;
  uint64_t elapsed_ns;
  const char *node;
} NodeRecord;

// ─────────────────────────────────────────────────────────────────────────────
// Growable buffers
// ─────────────────────────────────────────────────────────────────────────────

#define DEFINE_BUF(Name, T)                                                    \
  typedef struct {                                                             \
    T *data;                                                                   \
    size_t size, cap;                                                          \
  } Name;                                                                      \
  static void Name##_push(Name *b, T r) {                                      \
    if (b->size == b->cap) {                                                   \
      b->cap = b->cap ? b->cap * 2 : 64;                                       \
      b->data = realloc(b->data, b->cap * sizeof(T));                          \
    }                                                                          \
    b->data[b->size++] = r;                                                    \
  }

DEFINE_BUF(XferBuf, XferRecord)
DEFINE_BUF(LaunchBuf, LaunchRecord)
DEFINE_BUF(CopyBuf, CopyRecord)
DEFINE_BUF(LoadBuf, LoadRecord)
DEFINE_BUF(CompactBuf, CompactRecord)
DEFINE_BUF(HostBuf, HostRecord)
DEFINE_BUF(NodeBuf, NodeRecord)

static XferBuf g_scatter = {NULL, 0, 0};
static XferBuf g_gather = {NULL, 0, 0};
static LaunchBuf g_launch = {NULL, 0, 0};
static LaunchBuf g_free = {NULL, 0, 0};
static LaunchBuf g_alloc = {NULL, 0, 0};
static LoadBuf g_load = {NULL, 0, 0};
static CopyBuf g_copy = {NULL, 0, 0};
static CompactBuf g_compact = {NULL, 0, 0};
static HostBuf g_host = {NULL, 0, 0};
static NodeBuf g_node = {NULL, 0, 0};
// The open node brackets, innermost last.
#define MAX_OPEN_NODES 16
static struct {
  const char *node;
  uint64_t start_ns;
} g_open[MAX_OPEN_NODES];
static int g_n_open = 0;
static int g_iteration = 0;

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

uint64_t upmemrt_now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

void upmemrt_start_stat_collection(int iter) { g_iteration = iter; }

void upmemrt_record_scatter(uint64_t elapsed_ns, size_t bytes_per_dpu,
                            uint32_t num_dpus, size_t num_blocks,
                            const char *kind, const char *tag) {
  XferBuf_push(&g_scatter,
               (XferRecord){g_iteration, elapsed_ns, bytes_per_dpu, num_dpus,
                            num_blocks, kind, tag ? tag : ""});
}

void upmemrt_record_gather(uint64_t elapsed_ns, size_t bytes_per_dpu,
                           uint32_t num_dpus, size_t num_blocks,
                           const char *kind, const char *tag) {
  XferBuf_push(&g_gather,
               (XferRecord){g_iteration, elapsed_ns, bytes_per_dpu, num_dpus,
                            num_blocks, kind, tag ? tag : ""});
}

void upmemrt_record_launch(uint64_t elapsed_ns, uint32_t num_dpus) {
  LaunchBuf_push(&g_launch, (LaunchRecord){g_iteration, elapsed_ns, num_dpus});
}

void upmemrt_record_free(uint64_t elapsed_ns, uint32_t num_dpus) {
  LaunchBuf_push(&g_free, (LaunchRecord){g_iteration, elapsed_ns, num_dpus});
}

void upmemrt_record_alloc(uint64_t elapsed_ns, uint32_t num_dpus) {
  LaunchBuf_push(&g_alloc, (LaunchRecord){g_iteration, elapsed_ns, num_dpus});
}

void upmemrt_record_load(uint64_t elapsed_ns, uint32_t num_dpus) {
  LoadBuf_push(&g_load,
               (LoadRecord){g_iteration, elapsed_ns, num_dpus, g_n_open > 0});
}

void upmemrt_record_copy(uint64_t elapsed_ns, size_t bytes) {
  CopyBuf_push(&g_copy, (CopyRecord){g_iteration, elapsed_ns, bytes});
}

void upmemrt_record_compact(uint64_t elapsed_ns, size_t bytes,
                            const char *kind) {
  CompactBuf_push(&g_compact,
                  (CompactRecord){g_iteration, elapsed_ns, bytes, kind});
}

void upmemrt_record_host(uint64_t elapsed_ns, const char *function) {
  HostBuf_push(&g_host,
               (HostRecord){g_iteration, elapsed_ns, function ? function : ""});
}

void upmemrt_node_begin(const char *node) {
  if (g_n_open == MAX_OPEN_NODES) {
    fprintf(stderr, "upmemrt: node '%s' opens more than %d nested brackets\n",
            node, MAX_OPEN_NODES);
    abort();
  }
  g_open[g_n_open].node = node;
  g_open[g_n_open].start_ns = upmemrt_now_ns();
  g_n_open++;
}

void upmemrt_node_end(const char *node) {
  uint64_t now = upmemrt_now_ns();
  if (g_n_open == 0 || strcmp(g_open[g_n_open - 1].node, node) != 0) {
    fprintf(stderr, "upmemrt: node '%s' ends, but the open bracket is '%s'\n",
            node, g_n_open ? g_open[g_n_open - 1].node : "(none)");
    abort();
  }
  g_n_open--;
  NodeBuf_push(&g_node, (NodeRecord){g_iteration,
                                     now - g_open[g_n_open].start_ns, node});
}

// ─────────────────────────────────────────────────────────────────────────────
// CSV dump
// ─────────────────────────────────────────────────────────────────────────────

static void dump_xfer(const XferBuf *buf, const char *path) {
  FILE *f = fopen(path, "w");
  if (!f) {
    perror(path);
    return;
  }
  fprintf(f,
          "iteration,elapsed_ns,bytes_per_dpu,num_dpus,num_blocks,kind,tag\n");
  for (size_t i = 0; i < buf->size; i++) {
    const XferRecord *r = &buf->data[i];
    fprintf(f, "%d,%" PRIu64 ",%zu,%u,%zu,%s,%s\n", r->iteration, r->elapsed_ns,
            r->bytes_per_dpu, r->num_dpus, r->num_blocks, r->kind, r->tag);
  }
  fclose(f);
}

static void dump_launch(const LaunchBuf *buf, const char *path) {
  FILE *f = fopen(path, "w");
  if (!f) {
    perror(path);
    return;
  }
  fprintf(f, "iteration,elapsed_ns,num_dpus\n");
  for (size_t i = 0; i < buf->size; i++) {
    const LaunchRecord *r = &buf->data[i];
    fprintf(f, "%d,%" PRIu64 ",%u\n", r->iteration, r->elapsed_ns, r->num_dpus);
  }
  fclose(f);
}

static void dump_load(const LoadBuf *buf, const char *path) {
  FILE *f = fopen(path, "w");
  if (!f) {
    perror(path);
    return;
  }
  fprintf(f, "iteration,elapsed_ns,num_dpus,in_node\n");
  for (size_t i = 0; i < buf->size; i++) {
    const LoadRecord *r = &buf->data[i];
    fprintf(f, "%d,%" PRIu64 ",%u,%d\n", r->iteration, r->elapsed_ns,
            r->num_dpus, r->in_node);
  }
  fclose(f);
}

static void dump_copy(const CopyBuf *buf, const char *path) {
  FILE *f = fopen(path, "w");
  if (!f) {
    perror(path);
    return;
  }
  fprintf(f, "iteration,elapsed_ns,bytes\n");
  for (size_t i = 0; i < buf->size; i++) {
    const CopyRecord *r = &buf->data[i];
    fprintf(f, "%d,%" PRIu64 ",%zu\n", r->iteration, r->elapsed_ns, r->bytes);
  }
  fclose(f);
}

static void dump_compact(const CompactBuf *buf, const char *path) {
  FILE *f = fopen(path, "w");
  if (!f) {
    perror(path);
    return;
  }
  fprintf(f, "iteration,elapsed_ns,bytes,kind\n");
  for (size_t i = 0; i < buf->size; i++) {
    const CompactRecord *r = &buf->data[i];
    fprintf(f, "%d,%" PRIu64 ",%zu,%s\n", r->iteration, r->elapsed_ns, r->bytes,
            r->kind ? r->kind : "");
  }
  fclose(f);
}

static void dump_host(const HostBuf *buf, const char *path) {
  FILE *f = fopen(path, "w");
  if (!f) {
    perror(path);
    return;
  }
  fprintf(f, "iteration,elapsed_ns,function\n");
  for (size_t i = 0; i < buf->size; i++) {
    const HostRecord *r = &buf->data[i];
    fprintf(f, "%d,%" PRIu64 ",%s\n", r->iteration, r->elapsed_ns, r->function);
  }
  fclose(f);
}

static void dump_node(const NodeBuf *buf, const char *path) {
  FILE *f = fopen(path, "w");
  if (!f) {
    perror(path);
    return;
  }
  fprintf(f, "iteration,elapsed_ns,node\n");
  for (size_t i = 0; i < buf->size; i++) {
    const NodeRecord *r = &buf->data[i];
    fprintf(f, "%d,%" PRIu64 ",%s\n", r->iteration, r->elapsed_ns, r->node);
  }
  fclose(f);
}

void upmemrt_dump_stats(const char *prefix) {
  char path[4096];
  snprintf(path, sizeof(path), "%s_scatter.csv", prefix);
  dump_xfer(&g_scatter, path);
  snprintf(path, sizeof(path), "%s_gather.csv", prefix);
  dump_xfer(&g_gather, path);
  snprintf(path, sizeof(path), "%s_launch.csv", prefix);
  dump_launch(&g_launch, path);
  snprintf(path, sizeof(path), "%s_free.csv", prefix);
  dump_launch(&g_free, path);
  snprintf(path, sizeof(path), "%s_alloc.csv", prefix);
  dump_launch(&g_alloc, path);
  snprintf(path, sizeof(path), "%s_load.csv", prefix);
  dump_load(&g_load, path);
  snprintf(path, sizeof(path), "%s_copy.csv", prefix);
  dump_copy(&g_copy, path);
  snprintf(path, sizeof(path), "%s_compact.csv", prefix);
  dump_compact(&g_compact, path);
  snprintf(path, sizeof(path), "%s_host.csv", prefix);
  dump_host(&g_host, path);
  snprintf(path, sizeof(path), "%s_node.csv", prefix);
  dump_node(&g_node, path);
}

#else // UPMEM_RT_STATS not defined: the generated code still calls these.

#include "timers.h"

void upmemrt_node_begin(const char *node) { (void)node; }
void upmemrt_node_end(const char *node) { (void)node; }

#endif // UPMEM_RT_STATS
