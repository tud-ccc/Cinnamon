#include "aimc_dequeue.hh"
#include "aimc_intrinsics.hh"
#include "aimc_queue.hh"
#include "alpine_runtime.h"
#include <stdint.h>

static uint64_t g_next_tile = 0;
static const int kNumCoresVirt = 8;

static inline int tileToTid(uint64_t tile) {
  return (int)(tile % kNumCoresVirt);
}

extern "C" uint64_t alpine_alloc_tile(uint64_t height, uint64_t width) {
  (void)height;
  (void)width;
  return g_next_tile++;
}

extern "C" void alpine_write_weights(uint64_t tile, void *base, int64_t rows,
                                     int64_t cols, int64_t stride, int64_t x0,
                                     int64_t y0) {
  (void)stride;
  int tid = tileToTid(tile);
  const int8_t *W = (const int8_t *)base;
  for (int64_t y = 0; y < rows; ++y)
    for (int64_t x = 0; x < cols; ++x)
      aimcParamWrite((uint64_t)(x0 + x), (uint64_t)(y0 + y),
                     (uint64_t)(uint8_t)W[y * cols + x], tid);
}

extern "C" void alpine_enqueue_vec(uint64_t tile, void *base, int64_t len,
                                   int64_t stride, int64_t offset) {
  (void)stride;
  int tid = tileToTid(tile);
  int8_t *ptr = (int8_t *)base + offset;
  queueVector((int)len, ptr, tid);
}

extern "C" void alpine_dequeue_vec(uint64_t tile, void *base, int64_t len,
                                   int64_t stride, int64_t offset) {
  (void)stride;
  int tid = tileToTid(tile);
  int8_t *ptr = (int8_t *)base + offset;
  dequeueVector((int)len, ptr, tid);
}

extern "C" void alpine_mvm(uint64_t tile, void *in_base, int64_t in_len,
                           int64_t in_stride, int64_t in_off, void *out_base,
                           int64_t out_len, int64_t out_stride,
                           int64_t out_off) {
  (void)in_base;
  (void)in_len;
  (void)in_stride;
  (void)in_off;
  (void)out_stride;
  int tid = tileToTid(tile);
  aimcProcess(tid);
  if (out_base && out_len > 0) {
    int8_t *out = (int8_t *)out_base + out_off;
    dequeueVector((int)out_len, out, tid);
  }
}

extern "C" void alpine_process(uint64_t tile) {
  int tid = tileToTid(tile);
  aimcProcess(tid);
}
