#include "aimc.hh"
#include "aimc_quantize.hh"
#include "runtime_shims.hh"

extern "C" {

// --- tiny debug I/O (no libc) ---
static inline void dbg_write(const char *s, unsigned long n) {
  register long x0 __asm__("x0") = 1;
  register const char *x1 __asm__("x1") = s;
  register unsigned long x2 __asm__("x2") = n;
  register long x8 __asm__("x8") = 64;
  __asm__ volatile("svc #0" ::"r"(x0), "r"(x1), "r"(x2), "r"(x8) : "memory");
}
static int dbg_itoa(long v, char *out) {
  char tmp[32]; int t=0; unsigned long x;
  if (v < 0) { *out++='-'; x=(unsigned long)(-v); }
  else x=(unsigned long)v;
  if (x==0) { *out++='0'; return (int)1 + (v<0); }
  while (x && t < 32) { tmp[t++] = (char)('0' + (x % 10)); x/=10; }
  for (int i=t-1;i>=0;--i) *out++=tmp[i];
  return (int)t + (v<0);
}

static inline int64_t normalize_stride(int64_t stride, int64_t fallback) {
  return stride == 0 ? fallback : stride;
}

static inline int tileToTid(uint64_t tile) {
  static const int kNumCoresVirt = 8;
  return static_cast<int>(tile % kNumCoresVirt);
}

int32_t alpine_alloc_tile(uint64_t height, uint64_t width) {
#if defined(USE_CHECKER)
  extern void aimcSetArrayDimsRuntime(int, int);
  aimcSetArrayDimsRuntime(static_cast<int>(height), static_cast<int>(width));
#endif
  return 0;
}

// Expanded-ABI variant as declared in IR (extra mask and reserved args exist).
extern "C" void alpine_write_weights(uint64_t tile, void *alloc, void *aligned,
                                     int64_t offset, int64_t rows,
                                     int64_t cols, int64_t stride0,
                                     int64_t stride1, int64_t /*reserved*/) {
  (void)alloc;
  if (rows <= 0 || cols <= 0)
    return;

  const int8_t *data = static_cast<const int8_t *>(aligned);
  if (!data)
    return;

  const int tid = tileToTid(tile);
  const int64_t rowStride = normalize_stride(stride0, cols);
  const int64_t colStride = normalize_stride(stride1, 1);
  const int8_t *base = data + offset;

  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < cols; ++c) {
      const int64_t idx = r * rowStride + c * colStride;
      const int8_t val = base[idx];
      getAimc().aimcParamWrite(tid, static_cast<int>(c),
                               static_cast<int>(r), val);
    }
  }
}

void alpine_enqueue_vec(uint64_t tile, const void *alloc, const void * /*unused*/,
                        uint64_t offset, uint64_t size, uint64_t stride,
                        uint64_t /*reserved*/) {
  if (size == 0)
    return;

  const int8_t *aligned = static_cast<const int8_t *>(alloc);
  if (!aligned)
    return;

  const uint64_t step = (stride == 0) ? 1ULL : stride;
  const int tid = tileToTid(tile);

  const int8_t *start = aligned + offset;
  const int count = static_cast<int>(size);

  if (step == 1) {
    getAimc().writeInputVector(tid, start, count);
    return;
  }

  constexpr uint64_t kScratchCap = 512;
  int8_t stackBuf[kScratchCap];
  int8_t *buf = stackBuf;
  bool useHeap = false;
  if (size > kScratchCap) {
    buf = static_cast<int8_t *>(malloc(size));
    if (!buf)
      return;
    useHeap = true;
  }
  for (uint64_t i = 0; i < size; ++i)
    buf[i] = start[i * step];
  getAimc().writeInputVector(tid, buf, count);
  if (useHeap)
    free(buf);
}

void alpine_process(uint32_t tile) { aimcProcess(tileToTid(tile)); }

void alpine_dequeue_vec(uint64_t tile, void *alloc, const void * /*unused*/,
                        uint64_t offset, uint64_t size, uint64_t stride,
                        uint64_t /*reserved*/) {
  if (size == 0)
    return;

  int8_t *aligned = static_cast<int8_t *>(alloc);
  if (!aligned)
    return;

  const uint64_t step = (stride == 0) ? 1ULL : stride;
  const int tid = tileToTid(tile);
  const int count = static_cast<int>(size);

  constexpr uint64_t kScratchCap = 512;
  int8_t stackBuf[kScratchCap];
  int8_t *buf = stackBuf;
  bool useHeap = false;
  if (size > kScratchCap) {
    buf = static_cast<int8_t *>(malloc(size));
    if (!buf)
      return;
    useHeap = true;
  }

  getAimc().readOutputVector(tid, buf, count);
  for (uint64_t i = 0; i < size; ++i)
    aligned[offset + i * step] = buf[i];

  if (useHeap)
    free(buf);
}

void alpine_mvm(uint64_t /*tile*/, const void *in, const void * /*unused*/,
                uint64_t in_len, uint64_t /*stride*/, uint64_t in_off,
                void *out, const void * /*unused*/, uint64_t out_len,
                uint64_t /*stride*/, uint64_t out_off) {
  const int8_t *in_base = static_cast<const int8_t *>(in);
  int8_t *out_base = static_cast<int8_t *>(out);
  queueVector(static_cast<int>(in_len), const_cast<int8_t *>(in_base + in_off));
  aimcProcess();
  dequeueVector(static_cast<int>(out_len), out_base + out_off);
}

// Expanded memref ABI: r1
void alpine_quantize_r1(void *srcAlloc, void *srcAlign, int64_t srcOff,
                        int64_t srcSize, int64_t srcStride, void *dstAlloc,
                        void *dstAlign, int64_t dstOff, int64_t dstSize,
                        int64_t dstStride, float scale, int32_t zero) {
  (void)srcAlloc;
  (void)dstAlloc;
  if (srcSize <= 0 || dstSize <= 0)
    return;

  const float *src = static_cast<const float *>(srcAlign);
  int8_t *dst = static_cast<int8_t *>(dstAlign);
  if (!src || !dst)
    return;

  const int64_t count = srcSize < dstSize ? srcSize : dstSize;
  const int64_t sStride = normalize_stride(srcStride, 1);
  const int64_t dStride = normalize_stride(dstStride, 1);
  const float *srcBase = src + srcOff;
  int8_t *dstBase = dst + dstOff;
  const int8_t zeroPt = static_cast<int8_t>(zero);

  for (int64_t i = 0; i < count; ++i) {
    const float v = srcBase[i * sStride];
    dstBase[i * dStride] = aimc_quantize_scalar(v, scale, zeroPt);
  }
}

// Expanded memref ABI: r2
void alpine_quantize_r2(void *srcAlloc, void *srcAlign, int64_t srcOff,
                        int64_t rows, int64_t cols, int64_t srcStride0,
                        int64_t srcStride1, void *dstAlloc, void *dstAlign,
                        int64_t dstOff, int64_t dstRows, int64_t dstCols,
                        int64_t dstStride0, int64_t dstStride1, float scale,
                        int32_t zero) {
  (void)srcAlloc;
  (void)dstAlloc;
  if (rows <= 0 || cols <= 0 || dstRows <= 0 || dstCols <= 0)
    return;

  const float *src = static_cast<const float *>(srcAlign);
  int8_t *dst = static_cast<int8_t *>(dstAlign);
  if (!src || !dst)
    return;

  const int64_t rCount = rows < dstRows ? rows : dstRows;
  const int64_t cCount = cols < dstCols ? cols : dstCols;
  const int64_t sStride0 = normalize_stride(srcStride0, cols);
  const int64_t sStride1 = normalize_stride(srcStride1, 1);
  const int64_t dStride0 = normalize_stride(dstStride0, dstCols);
  const int64_t dStride1 = normalize_stride(dstStride1, 1);
  const float *srcBase = src + srcOff;
  int8_t *dstBase = dst + dstOff;
  const int8_t zeroPt = static_cast<int8_t>(zero);

  for (int64_t r = 0; r < rCount; ++r) {
    for (int64_t c = 0; c < cCount; ++c) {
      const float v = srcBase[r * sStride0 + c * sStride1];
      dstBase[r * dStride0 + c * dStride1] =
          aimc_quantize_scalar(v, scale, zeroPt);
    }
  }
}

// Expanded memref ABI: deq r1
void alpine_dequantize_r1(void *srcAlloc, void *srcAlign, int64_t srcOff,
                          int64_t srcSize, int64_t srcStride,
                          void *dstAlloc, void *dstAlign, int64_t dstOff,
                          int64_t dstSize, int64_t dstStride, float scale,
                          int32_t zero) {
  (void)srcAlloc;
  (void)dstAlloc;
  if (srcSize <= 0 || dstSize <= 0)
    return;

  const int8_t *src = static_cast<const int8_t *>(srcAlign);
  float *dst = static_cast<float *>(dstAlign);
  if (!src || !dst)
    return;

  const int64_t count = srcSize < dstSize ? srcSize : dstSize;
  const int64_t sStride = normalize_stride(srcStride, 1);
  const int64_t dStride = normalize_stride(dstStride, 1);
  const int8_t zeroPt = static_cast<int8_t>(zero);
  const int8_t *srcBase = src + srcOff;
  float *dstBase = dst + dstOff;

  for (int64_t i = 0; i < count; ++i) {
    const int8_t q = srcBase[i * sStride];
    dstBase[i * dStride] = aimc_dequantize_scalar(q, scale, zeroPt);
  }
}

void alpine_relu_r1(void *srcAlloc, void *srcAlign, int64_t srcOff,
                    int64_t srcSize, int64_t srcStride, void *dstAlloc,
                    void *dstAlign, int64_t dstOff, int64_t dstSize,
                    int64_t dstStride) {
  (void)srcAlloc;
  (void)dstAlloc;
  if (srcSize <= 0 || dstSize <= 0)
    return;

  const float *src = static_cast<const float *>(srcAlign);
  float *dst = static_cast<float *>(dstAlign);
  if (!src || !dst)
    return;

  const int64_t count = srcSize < dstSize ? srcSize : dstSize;
  const int64_t sStride = normalize_stride(srcStride, 1);
  const int64_t dStride = normalize_stride(dstStride, 1);
  const float *srcBase = src + srcOff;
  float *dstBase = dst + dstOff;

  for (int64_t i = 0; i < count; ++i) {
    const float v = srcBase[i * sStride];
    dstBase[i * dStride] = v > 0.0f ? v : 0.0f;
  }
}

void alpine_relu_r2(const float *src, float *dst, uint64_t rows,
                    uint64_t cols) {
  uint64_t n = rows * cols;
  for (uint64_t i = 0; i < n; ++i)
    dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
}

// Crossbar read helper sharing the same internal state as writers.
int8_t alpine_read_param(int x, int y) {
  return static_cast<int8_t>(getAimc().aimcParamRead(0, x, y));
}
int8_t alpine_read_input_at(int idx) { return getAimc().peekInput(0, idx); }
int8_t alpine_read_output_at(int idx) { return getAimc().peekOutput(0, idx); }


} // extern "C"
