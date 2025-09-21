// Minimal C API bridging MLIR-generated kernels to AIMC runtime helpers.
#pragma once
#include <stdint.h>

#ifndef _ALPINE_RUNTIME_
#define _ALPINE_RUNTIME_

#ifdef __cplusplus
extern "C" {
#endif

uint64_t alpine_alloc_tile(uint64_t height, uint64_t width);

void alpine_write_weights(uint64_t tile, const void *weights,
                          const void * /*unused*/, uint64_t height,
                          uint64_t width, uint64_t /*stride*/, uint64_t x0,
                          uint64_t y0);

void alpine_enqueue_vec(uint64_t tile, const void *in, const void * /*unused*/,
                        uint64_t len, uint64_t /*stride*/, uint64_t offset);

void alpine_process(uint64_t tile);

void alpine_dequeue_vec(uint64_t tile, void *out, const void * /*unused*/,
                        uint64_t len, uint64_t /*stride*/, uint64_t offset);

void alpine_mvm(uint64_t tile, const void *in, const void * /*unused*/,
                uint64_t in_len, uint64_t /*stride*/, uint64_t in_off,
                void *out, const void * /*unused*/, uint64_t out_len,
                uint64_t /*stride*/, uint64_t out_off);

void alpine_quantize_r1(const float *src, int8_t *dst, uint64_t len,
                        float scale, int32_t zero);
void alpine_quantize_r2(const float *src, int8_t *dst, uint64_t rows,
                        uint64_t cols, float scale, int32_t zero);

void alpine_dequantize_r1(const int8_t *src, float *dst, uint64_t len,
                          float scale, int32_t zero);
void alpine_dequantize_r2(const int8_t *src, float *dst, uint64_t rows,
                          uint64_t cols, float scale, int32_t zero);

void alpine_relu_r1(const float *src, float *dst, uint64_t len);
void alpine_relu_r2(const float *src, float *dst, uint64_t rows, uint64_t cols);

#ifdef __cplusplus
}
#endif

#endif