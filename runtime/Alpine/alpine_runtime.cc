#include "aimc.hh"
#include "aimc_quantize.hh"

extern "C" {

uint64_t alpine_alloc_tile(uint64_t height, uint64_t width) {
#if defined(USE_CHECKER)
  extern void aimcSetArrayDimsRuntime(int, int);
  aimcSetArrayDimsRuntime(static_cast<int>(height), static_cast<int>(width));
#endif
  return 0ULL;
}

void alpine_write_weights(uint64_t /*tile*/, const void *weights,
                          const void * /*unused*/, uint64_t height,
                          uint64_t width, uint64_t /*stride*/, uint64_t x0,
                          uint64_t y0) {
  const int8_t *W = static_cast<const int8_t *>(weights);
  mapMatrix(static_cast<int>(x0), static_cast<int>(y0),
            static_cast<int>(height), static_cast<int>(width),
            const_cast<int8_t *>(W));
}

void alpine_enqueue_vec(uint64_t /*tile*/, const void *in,
                        const void * /*unused*/, uint64_t len,
                        uint64_t /*stride*/, uint64_t offset) {
  const int8_t *base = static_cast<const int8_t *>(in);
  queueVector(static_cast<int>(len), const_cast<int8_t *>(base + offset));
}

void alpine_process(uint64_t /*tile*/) { aimcProcess(); }

void alpine_dequeue_vec(uint64_t /*tile*/, void *out, const void * /*unused*/,
                        uint64_t len, uint64_t /*stride*/, uint64_t offset) {
  int8_t *base = static_cast<int8_t *>(out);
  dequeueVector(static_cast<int>(len), base + offset);
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

void alpine_quantize_r1(const float *src, int8_t *dst, uint64_t len,
                        float scale, int32_t zero) {
  aimc_quantize_vector(static_cast<int>(len), src, dst, scale,
                       static_cast<int8_t>(zero));
}

void alpine_quantize_r2(const float *src, int8_t *dst, uint64_t rows,
                        uint64_t cols, float scale, int32_t zero) {
  uint64_t n = rows * cols;
  aimc_quantize_vector(static_cast<int>(n), src, dst, scale,
                       static_cast<int8_t>(zero));
}

void alpine_dequantize_r1(const int8_t *src, float *dst, uint64_t len,
                          float scale, int32_t zero) {
  aimc_dequantize_vector(static_cast<int>(len), src, dst, scale,
                         static_cast<int8_t>(zero));
}

void alpine_dequantize_r2(const int8_t *src, float *dst, uint64_t rows,
                          uint64_t cols, float scale, int32_t zero) {
  uint64_t n = rows * cols;
  aimc_dequantize_vector(static_cast<int>(n), src, dst, scale,
                         static_cast<int8_t>(zero));
}

void alpine_relu_r1(const float *src, float *dst, uint64_t len) {
  for (uint64_t i = 0; i < len; ++i)
    dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
}

void alpine_relu_r2(const float *src, float *dst, uint64_t rows,
                    uint64_t cols) {
  uint64_t n = rows * cols;
  for (uint64_t i = 0; i < n; ++i)
    dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
}

} // extern "C"
