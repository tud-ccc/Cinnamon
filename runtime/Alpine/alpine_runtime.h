// Minimal C API bridging MLIR-generated kernels to AIMC runtime helpers.
#pragma once
#include <stdint.h>

#ifndef _ALPINE_RUNTIME_
#define _ALPINE_RUNTIME_

#ifdef __cplusplus
extern "C" {
#endif

int32_t alpine_alloc_tile(uint64_t height, uint64_t width);

void alpine_write_weights(uint64_t tile, void *alloc, void *aligned,
                          int64_t offset, int64_t rows, int64_t cols,
                          int64_t stride0, int64_t stride1, int64_t reserved);

void alpine_enqueue_vec(uint64_t tile, const void *alloc, const void *unused,
                        uint64_t offset, uint64_t size, uint64_t stride,
                        uint64_t reserved);

void alpine_process(uint32_t tile);

void alpine_dequeue_vec(uint64_t tile, void *alloc, const void *unused,
                        uint64_t offset, uint64_t size, uint64_t stride,
                        uint64_t reserved);

void alpine_mvm(uint64_t tile, const void *in, const void * /*unused*/,
                uint64_t in_len, uint64_t /*stride*/, uint64_t in_off,
                void *out, const void * /*unused*/, uint64_t out_len,
                uint64_t /*stride*/, uint64_t out_off);

void alpine_quantize_r1(void *srcAlloc, void *srcAlign, int64_t srcOff,
                        int64_t srcSize, int64_t srcStride, void *dstAlloc,
                        void *dstAlign, int64_t dstOff, int64_t dstSize,
                        int64_t dstStride, float scale, int32_t zero);
void alpine_quantize_r2(void *srcAlloc, void *srcAlign, int64_t srcOff,
                        int64_t rows, int64_t cols, int64_t srcStride0,
                        int64_t srcStride1, void *dstAlloc, void *dstAlign,
                        int64_t dstOff, int64_t dstRows, int64_t dstCols,
                        int64_t dstStride0, int64_t dstStride1, float scale,
                        int32_t zero);

void alpine_dequantize_r1(void *srcAlloc, void *srcAlign, int64_t srcOff,
                          int64_t srcSize, int64_t srcStride, void *dstAlloc,
                          void *dstAlign, int64_t dstOff, int64_t dstSize,
                          int64_t dstStride, float scale, int32_t zero);

void alpine_relu_r1(void *srcAlloc, void *srcAlign, int64_t srcOff,
                    int64_t srcSize, int64_t srcStride, void *dstAlloc,
                    void *dstAlign, int64_t dstOff, int64_t dstSize,
                    int64_t dstStride);
void alpine_relu_r2(const float *src, float *dst, uint64_t rows, uint64_t cols);

// Helper to read a parameter value from the emulated crossbar using the
// same underlying state as the runtime (avoids per-TU state duplication).
int8_t alpine_read_param(int x, int y);
int8_t alpine_read_input_at(int idx);
int8_t alpine_read_output_at(int idx);

#ifdef __cplusplus
}
#endif

#endif
