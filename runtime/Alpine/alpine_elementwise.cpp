#include "alpine_memref_abi.h"
#include <stdint.h>

static inline int8_t clamp_i8(int v) {
  if (v < -128)
    return -128;
  if (v > 127)
    return 127;
  return (int8_t)v;
}

static inline int rint_away_from_zero(float x) {
  return (int)(x >= 0.0f ? (x + 0.5f) : (x - 0.5f));
}

extern "C" void alpine_relu_r1(void *src, void *dst) {
  auto *s = (StridedMemRefType<float, 1> *)src;
  auto *d = (StridedMemRefType<float, 1> *)dst;
  foreachIndex<float, 1>(s, [&](const int64_t i[1]) {
    float v = memrefAt(s, i);
    memrefAt(d, i) = v > 0.f ? v : 0.f;
  });
}

extern "C" void alpine_relu_r2(void *src, void *dst) {
  auto *s = (StridedMemRefType<float, 2> *)src;
  auto *d = (StridedMemRefType<float, 2> *)dst;
  foreachIndex<float, 2>(s, [&](const int64_t i[2]) {
    float v = memrefAt(s, i);
    memrefAt(d, i) = v > 0.f ? v : 0.f;
  });
}

extern "C" void alpine_quantize_r1(void *src, void *dst, float scale,
                                   int32_t zero) {
  auto *s = (StridedMemRefType<float, 1> *)src;
  auto *d = (StridedMemRefType<int8_t, 1> *)dst;
  if (scale == 0.0f) {
    foreachIndex<float, 1>(
        s, [&](const int64_t i[1]) { memrefAt(d, i) = (int8_t)zero; });
    return;
  }
  foreachIndex<float, 1>(s, [&](const int64_t i[1]) {
    int q = rint_away_from_zero(memrefAt(s, i) / scale) + zero;
    memrefAt(d, i) = clamp_i8(q);
  });
}

extern "C" void alpine_quantize_r2(void *src, void *dst, float scale,
                                   int32_t zero) {
  auto *s = (StridedMemRefType<float, 2> *)src;
  auto *d = (StridedMemRefType<int8_t, 2> *)dst;
  if (scale == 0.0f) {
    foreachIndex<float, 2>(
        s, [&](const int64_t i[2]) { memrefAt(d, i) = (int8_t)zero; });
    return;
  }
  foreachIndex<float, 2>(s, [&](const int64_t i[2]) {
    int q = rint_away_from_zero(memrefAt(s, i) / scale) + zero;
    memrefAt(d, i) = clamp_i8(q);
  });
}

extern "C" void alpine_dequantize_r1(void *src, void *dst, float scale,
                                     int32_t zero) {
  auto *s = (StridedMemRefType<int8_t, 1> *)src;
  auto *d = (StridedMemRefType<float, 1> *)dst;
  foreachIndex<int8_t, 1>(s, [&](const int64_t i[1]) {
    int qi = (int)memrefAt(s, i);
    memrefAt(d, i) = (float)(qi - zero) * scale;
  });
}

extern "C" void alpine_dequantize_r2(void *src, void *dst, float scale,
                                     int32_t zero) {
  auto *s = (StridedMemRefType<int8_t, 2> *)src;
  auto *d = (StridedMemRefType<float, 2> *)dst;
  foreachIndex<int8_t, 2>(s, [&](const int64_t i[2]) {
    int qi = (int)memrefAt(s, i);
    memrefAt(d, i) = (float)(qi - zero) * scale;
  });
}
