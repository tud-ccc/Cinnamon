#ifdef UPMEM_RT_STATS

#include "timers.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
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

static XferBuf g_scatter = {NULL, 0, 0};
static XferBuf g_gather = {NULL, 0, 0};
static LaunchBuf g_launch = {NULL, 0, 0};
static LaunchBuf g_free = {NULL, 0, 0};
static LaunchBuf g_alloc = {NULL, 0, 0};
static CopyBuf g_copy = {NULL, 0, 0};
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

void upmemrt_record_copy(uint64_t elapsed_ns, size_t bytes) {
  CopyBuf_push(&g_copy, (CopyRecord){g_iteration, elapsed_ns, bytes});
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
  snprintf(path, sizeof(path), "%s_copy.csv", prefix);
  dump_copy(&g_copy, path);
}

#endif // UPMEM_RT_STATS
