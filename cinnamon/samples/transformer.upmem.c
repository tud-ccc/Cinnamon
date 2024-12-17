// UPMEM-TRANSLATE: COMPILE_forward:8:forward;COMPILE_forward_3:8:forward_3;COMPILE_forward_6:8:forward_6;COMPILE_forward_8:16:forward_8;COMPILE_mha:8:mha;COMPILE_mha_9:8:mha_9;COMPILE_rmsnorm:16:rmsnorm;COMPILE_rmsnorm_11:16:rmsnorm_11;COMPILE_softmax:16:softmax;COMPILE_softmax_13:16:softmax_13;COMPILE_rmsnorm_1048576:16:rmsnorm_1048576;COMPILE_rmsnorm_1048576_14:16:rmsnorm_1048576_14;COMPILE_softmax_1048576:16:softmax_1048576;COMPILE_softmax_1048576_16:16:softmax_1048576_16;COMPILE_va_1048576:16:va_1048576;COMPILE_rmsnorm_262144:16:rmsnorm_262144;COMPILE_rmsnorm_262144_17:16:rmsnorm_262144_17;COMPILE_softmax_262144:16:softmax_262144;COMPILE_softmax_262144_19:16:softmax_262144_19;COMPILE_rmsnorm_262144_opt:16:rmsnorm_262144_opt;COMPILE_rmsnorm_262144_opt_20:16:rmsnorm_262144_opt_20;COMPILE_softmax_262144_opt:16:softmax_262144_opt;COMPILE_softmax_262144_opt_21:16:softmax_262144_opt_21;COMPILE_va_262144:16:va_262144;COMPILE_mha_big:16:mha_big;COMPILE_mha_big_22:16:mha_big_22;COMPILE_mha_big_23:16:mha_big_23;COMPILE_mha_big_24:16:mha_big_24;COMPILE_mha_big_25:16:mha_big_25;COMPILE_test_0:16:test_0;COMPILE_test_2:16:test_2;

#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <perfcounter.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "expf.c"

#ifdef COMPILE_forward
void forward() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = me();
  int32_t v3 = v2 * 1152;
  int32_t v4 = v1 + v3;
  __dma_aligned float v5[288];
  int32_t v6 = v1 + 9216;
  int32_t v7 = v6 + v3;
  __dma_aligned float v8[288];
  int32_t v9 = v6 + 9216;
  int32_t v10 = v2 * 8;
  int32_t v11 = v9 + v10;
  __dma_aligned float v12[2];
  mram_read((const __mram_ptr float*)v4, (float*)v5, 288 * sizeof(float));
  mram_read((const __mram_ptr float*)v7, (float*)v8, 288 * sizeof(float));
  for (int32_t v13 = 0; v13 < 288; v13 += 1) {
    float v14 = v5[v13];
    float v15 = v8[v13];
    float v16 = v12[0];
    float v17 = v14 * v15;
    float v18 = v17 + v16;
    v12[0] = v18;
  }
  mram_write((const float*)v12, (__mram_ptr float*)v11, 2 * sizeof(float));
  return;
}
#endif

#ifdef COMPILE_forward_3
void forward_3() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = me();
  int32_t v3 = v2 * 24;
  int32_t v4 = v1 + v3;
  __dma_aligned float v5[6];
  int32_t v6 = v1 + 192;
  int32_t v7 = v6 + v3;
  __dma_aligned float v8[6];
  int32_t v9 = v6 + 192;
  int32_t v10 = v9 + v3;
  __dma_aligned float v11[6];
  mram_read((const __mram_ptr float*)v4, (float*)v5, 6 * sizeof(float));
  mram_read((const __mram_ptr float*)v7, (float*)v8, 6 * sizeof(float));
  for (int32_t v12 = 0; v12 < 6; v12 += 1) {
    float v13 = v5[v12];
    float v14 = v8[v12];
    float v15 = v13 + v14;
    v11[v12] = v15;
  }
  mram_write((const float*)v11, (__mram_ptr float*)v10, 6 * sizeof(float));
  return;
}
#endif

#ifdef COMPILE_forward_6
void forward_6() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = me();
  int32_t v3 = v2 * 3072;
  int32_t v4 = v1 + v3;
  __dma_aligned float v5[768];
  int32_t v6 = v1 + 24576;
  int32_t v7 = v6 + v3;
  __dma_aligned float v8[768];
  int32_t v9 = v6 + 24576;
  int32_t v10 = v2 * 8;
  int32_t v11 = v9 + v10;
  __dma_aligned float v12[2];
  mram_read((const __mram_ptr float*)v4, (float*)v5, 512 * sizeof(float));
  mram_read((const __mram_ptr float*)v4 + 512, (float*)v5 + 512, 256 * sizeof(float));
  mram_read((const __mram_ptr float*)v7, (float*)v8, 512 * sizeof(float));
  mram_read((const __mram_ptr float*)v7 + 512, (float*)v8 + 512, 256 * sizeof(float));
  for (int32_t v13 = 0; v13 < 768; v13 += 1) {
    float v14 = v5[v13];
    float v15 = v8[v13];
    float v16 = v12[0];
    float v17 = v14 * v15;
    float v18 = v17 + v16;
    v12[0] = v18;
  }
  mram_write((const float*)v12, (__mram_ptr float*)v11, 2 * sizeof(float));
  return;
}
#endif

#ifdef COMPILE_forward_8
void forward_8() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = me();
  int32_t v3 = v2 * 1152;
  int32_t v4 = v1 + v3;
  __dma_aligned float v5[288];
  int32_t v6 = v1 + 18432;
  int32_t v7 = v6 + v3;
  __dma_aligned float v8[288];
  int32_t v9 = v6 + 18432;
  int32_t v10 = v2 * 8;
  int32_t v11 = v9 + v10;
  __dma_aligned float v12[2];
  mram_read((const __mram_ptr float*)v4, (float*)v5, 288 * sizeof(float));
  mram_read((const __mram_ptr float*)v7, (float*)v8, 288 * sizeof(float));
  for (int32_t v13 = 0; v13 < 288; v13 += 1) {
    float v14 = v5[v13];
    float v15 = v8[v13];
    float v16 = v12[0];
    float v17 = v14 * v15;
    float v18 = v17 + v16;
    v12[0] = v18;
  }
  mram_write((const float*)v12, (__mram_ptr float*)v11, 2 * sizeof(float));
  return;
}
#endif

#ifdef COMPILE_mha
void mha() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = me();
  int32_t v3 = v2 * 24;
  int32_t v4 = v1 + v3;
  __dma_aligned float v5[6];
  int32_t v6 = v1 + 192;
  int32_t v7 = v6 + v3;
  __dma_aligned float v8[6];
  int32_t v9 = v6 + 192;
  int32_t v10 = v9 + v3;
  __dma_aligned float v11[6];
  mram_read((const __mram_ptr float*)v4, (float*)v5, 6 * sizeof(float));
  mram_read((const __mram_ptr float*)v7, (float*)v8, 6 * sizeof(float));
  for (int32_t v12 = 0; v12 < 6; v12 += 1) {
    float v13 = v5[v12];
    float v14 = v8[v12];
    float v15 = v13 * v14;
    v11[v12] = v15;
  }
  mram_write((const float*)v11, (__mram_ptr float*)v10, 6 * sizeof(float));
  return;
}
#endif

#ifdef COMPILE_mha_9
void mha_9() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = me();
  int32_t v3 = v2 * 24;
  int32_t v4 = v1 + v3;
  __dma_aligned float v5[6];
  int32_t v6 = v1 + 192;
  __dma_aligned float v7[2];
  int32_t v8 = v6 + 32;
  int32_t v9 = v8 + v3;
  __dma_aligned float v10[6];
  mram_read((const __mram_ptr float*)v4, (float*)v5, 6 * sizeof(float));
  mram_read((const __mram_ptr float*)v6, (float*)v7, 2 * sizeof(float));
  for (int32_t v11 = 0; v11 < 6; v11 += 1) {
    float v12 = v5[v11];
    float v13 = v7[0];
    float v14 = v12 * v13;
    v10[v11] = v14;
  }
  mram_write((const float*)v10, (__mram_ptr float*)v9, 6 * sizeof(float));
  return;
}
#endif

#ifdef COMPILE_rmsnorm
void rmsnorm() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = me();
  int32_t v3 = v2 * 72;
  int32_t v4 = v1 + v3;
  __dma_aligned float v5[18];
  int32_t v6 = v1 + 1152;
  int32_t v7 = v6 + v3;
  __dma_aligned float v8[18];
  int32_t v9 = v6 + 1152;
  int32_t v10 = v9 + v3;
  __dma_aligned float v11[18];
  mram_read((const __mram_ptr float*)v4, (float*)v5, 18 * sizeof(float));
  mram_read((const __mram_ptr float*)v7, (float*)v8, 18 * sizeof(float));
  for (int32_t v12 = 0; v12 < 18; v12 += 1) {
    float v13 = v5[v12];
    float v14 = v8[v12];
    float v15 = v13 * v14;
    v11[v12] = v15;
  }
  mram_write((const float*)v11, (__mram_ptr float*)v10, 18 * sizeof(float));
  return;
}
#endif

#ifdef COMPILE_rmsnorm_11
void rmsnorm_11() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = me();
  int32_t v3 = v2 * 72;
  int32_t v4 = v1 + v3;
  __dma_aligned float v5[18];
  int32_t v6 = v1 + 1152;
  __dma_aligned float v7[2];
  int32_t v8 = v6 + 64;
  int32_t v9 = v8 + v3;
  __dma_aligned float v10[18];
  mram_read((const __mram_ptr float*)v4, (float*)v5, 18 * sizeof(float));
  mram_read((const __mram_ptr float*)v6, (float*)v7, 2 * sizeof(float));
  for (int32_t v11 = 0; v11 < 18; v11 += 1) {
    float v12 = v5[v11];
    float v13 = v7[0];
    float v14 = v12 * v13;
    v10[v11] = v14;
  }
  mram_write((const float*)v10, (__mram_ptr float*)v9, 18 * sizeof(float));
  return;
}
#endif

#ifdef COMPILE_softmax
void softmax() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = me();
  int32_t v3 = v2 * 8;
  int32_t v4 = v1 + v3;
  __dma_aligned float v5[2];
  int32_t v6 = v1 + 128;
  __dma_aligned float v7[2];
  int32_t v8 = v6 + 64;
  int32_t v9 = v8 + v3;
  __dma_aligned float v10[2];
  mram_read((const __mram_ptr float*)v4, (float*)v5, 2 * sizeof(float));
  mram_read((const __mram_ptr float*)v6, (float*)v7, 2 * sizeof(float));
  for (int32_t v11 = 0; v11 < 2; v11 += 1) {
    float v12 = v5[v11];
    float v13 = v7[0];
    float v14 = v12 - v13;
    v10[v11] = v14;
  }
  mram_write((const float*)v10, (__mram_ptr float*)v9, 2 * sizeof(float));
  return;
}
#endif

#ifdef COMPILE_softmax_13
void softmax_13() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = me();
  int32_t v3 = v2 * 8;
  int32_t v4 = v1 + v3;
  __dma_aligned float v5[2];
  int32_t v6 = v1 + 128;
  __dma_aligned float v7[2];
  int32_t v8 = v6 + 64;
  int32_t v9 = v8 + v3;
  __dma_aligned float v10[2];
  mram_read((const __mram_ptr float*)v4, (float*)v5, 2 * sizeof(float));
  mram_read((const __mram_ptr float*)v6, (float*)v7, 2 * sizeof(float));
  for (int32_t v11 = 0; v11 < 2; v11 += 1) {
    float v12 = v5[v11];
    float v13 = v7[0];
    float v14 = v12 / v13;
    v10[v11] = v14;
  }
  mram_write((const float*)v10, (__mram_ptr float*)v9, 2 * sizeof(float));
  return;
}
#endif

#ifdef COMPILE_rmsnorm_1048576
void rmsnorm_1048576() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = me();
  int32_t v3 = v2 * 1024;
  int32_t v4 = v1 + v3;
  __dma_aligned float v5[256];
  int32_t v6 = v1 + 16384;
  int32_t v7 = v6 + v3;
  __dma_aligned float v8[256];
  int32_t v9 = v6 + 16384;
  int32_t v10 = v9 + v3;
  __dma_aligned float v11[256];
  mram_read((const __mram_ptr float*)v4, (float*)v5, 256 * sizeof(float));
  mram_read((const __mram_ptr float*)v7, (float*)v8, 256 * sizeof(float));
  for (int32_t v12 = 0; v12 < 256; v12 += 1) {
    float v13 = v5[v12];
    float v14 = v8[v12];
    float v15 = v13 * v14;
    v11[v12] = v15;
  }
  mram_write((const float*)v11, (__mram_ptr float*)v10, 256 * sizeof(float));
  return;
}
#endif

#ifdef COMPILE_rmsnorm_1048576_14
void rmsnorm_1048576_14() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = me();
  int32_t v3 = v2 * 1024;
  int32_t v4 = v1 + v3;
  __dma_aligned float v5[256];
  int32_t v6 = v1 + 16384;
  __dma_aligned float v7[2];
  int32_t v8 = v6 + 64;
  int32_t v9 = v8 + v3;
  __dma_aligned float v10[256];
  mram_read((const __mram_ptr float*)v4, (float*)v5, 256 * sizeof(float));
  mram_read((const __mram_ptr float*)v6, (float*)v7, 2 * sizeof(float));
  for (int32_t v11 = 0; v11 < 256; v11 += 1) {
    float v12 = v5[v11];
    float v13 = v7[0];
    float v14 = v12 * v13;
    v10[v11] = v14;
  }
  mram_write((const float*)v10, (__mram_ptr float*)v9, 256 * sizeof(float));
  return;
}
#endif

#ifdef COMPILE_softmax_1048576
void softmax_1048576() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = me();
  int32_t v3 = v2 * 1024;
  int32_t v4 = v1 + v3;
  __dma_aligned float v5[256];
  int32_t v6 = v1 + 16384;
  __dma_aligned float v7[2];
  int32_t v8 = v6 + 64;
  int32_t v9 = v8 + v3;
  __dma_aligned float v10[256];
  mram_read((const __mram_ptr float*)v4, (float*)v5, 256 * sizeof(float));
  mram_read((const __mram_ptr float*)v6, (float*)v7, 2 * sizeof(float));
  for (int32_t v11 = 0; v11 < 256; v11 += 1) {
    float v12 = v5[v11];
    float v13 = v7[0];
    float v14 = v12 - v13;
    v10[v11] = v14;
  }
  mram_write((const float*)v10, (__mram_ptr float*)v9, 256 * sizeof(float));
  return;
}
#endif

#ifdef COMPILE_softmax_1048576_16
void softmax_1048576_16() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = me();
  int32_t v3 = v2 * 1024;
  int32_t v4 = v1 + v3;
  __dma_aligned float v5[256];
  int32_t v6 = v1 + 16384;
  __dma_aligned float v7[2];
  int32_t v8 = v6 + 64;
  int32_t v9 = v8 + v3;
  __dma_aligned float v10[256];
  mram_read((const __mram_ptr float*)v4, (float*)v5, 256 * sizeof(float));
  mram_read((const __mram_ptr float*)v6, (float*)v7, 2 * sizeof(float));
  for (int32_t v11 = 0; v11 < 256; v11 += 1) {
    float v12 = v5[v11];
    float v13 = v7[0];
    float v14 = v12 / v13;
    v10[v11] = v14;
  }
  mram_write((const float*)v10, (__mram_ptr float*)v9, 256 * sizeof(float));
  return;
}
#endif

#ifdef COMPILE_va_1048576
void va_1048576() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = me();
  int32_t v3 = v2 * 1024;
  int32_t v4 = v1 + v3;
  __dma_aligned float v5[256];
  int32_t v6 = v1 + 16384;
  int32_t v7 = v6 + v3;
  __dma_aligned float v8[256];
  int32_t v9 = v6 + 16384;
  int32_t v10 = v9 + v3;
  __dma_aligned float v11[256];
  mram_read((const __mram_ptr float*)v4, (float*)v5, 256 * sizeof(float));
  mram_read((const __mram_ptr float*)v7, (float*)v8, 256 * sizeof(float));
  for (int32_t v12 = 0; v12 < 256; v12 += 1) {
    float v13 = v5[v12];
    float v14 = v8[v12];
    float v15 = v13 + v14;
    v11[v12] = v15;
  }
  mram_write((const float*)v11, (__mram_ptr float*)v10, 256 * sizeof(float));
  return;
}
#endif

#ifdef COMPILE_rmsnorm_262144
void rmsnorm_262144() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = me();
  int32_t v3 = v2 * 256;
  int32_t v4 = v1 + v3;
  __dma_aligned float v5[64];
  int32_t v6 = v1 + 4096;
  int32_t v7 = v6 + v3;
  __dma_aligned float v8[64];
  int32_t v9 = v6 + 4096;
  int32_t v10 = v9 + v3;
  __dma_aligned float v11[64];
  mram_read((const __mram_ptr float*)v4, (float*)v5, 64 * sizeof(float));
  mram_read((const __mram_ptr float*)v7, (float*)v8, 64 * sizeof(float));
  for (int32_t v12 = 0; v12 < 64; v12 += 1) {
    float v13 = v5[v12];
    float v14 = v8[v12];
    float v15 = v13 * v14;
    v11[v12] = v15;
  }
  mram_write((const float*)v11, (__mram_ptr float*)v10, 64 * sizeof(float));
  return;
}
#endif

#ifdef COMPILE_rmsnorm_262144_17
void rmsnorm_262144_17() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = me();
  int32_t v3 = v2 * 256;
  int32_t v4 = v1 + v3;
  __dma_aligned float v5[64];
  int32_t v6 = v1 + 4096;
  __dma_aligned float v7[2];
  int32_t v8 = v6 + 64;
  int32_t v9 = v8 + v3;
  __dma_aligned float v10[64];
  mram_read((const __mram_ptr float*)v4, (float*)v5, 64 * sizeof(float));
  mram_read((const __mram_ptr float*)v6, (float*)v7, 2 * sizeof(float));
  for (int32_t v11 = 0; v11 < 64; v11 += 1) {
    float v12 = v5[v11];
    float v13 = v7[0];
    float v14 = v12 * v13;
    v10[v11] = v14;
  }
  mram_write((const float*)v10, (__mram_ptr float*)v9, 64 * sizeof(float));
  return;
}
#endif

#ifdef COMPILE_softmax_262144
void softmax_262144() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = me();
  int32_t v3 = v2 * 256;
  int32_t v4 = v1 + v3;
  __dma_aligned float v5[64];
  int32_t v6 = v1 + 4096;
  __dma_aligned float v7[2];
  int32_t v8 = v6 + 64;
  int32_t v9 = v8 + v3;
  __dma_aligned float v10[64];
  mram_read((const __mram_ptr float*)v4, (float*)v5, 64 * sizeof(float));
  mram_read((const __mram_ptr float*)v6, (float*)v7, 2 * sizeof(float));
  for (int32_t v11 = 0; v11 < 64; v11 += 1) {
    float v12 = v5[v11];
    float v13 = v7[0];
    float v14 = v12 - v13;
    v10[v11] = v14;
  }
  mram_write((const float*)v10, (__mram_ptr float*)v9, 64 * sizeof(float));
  return;
}
#endif

#ifdef COMPILE_softmax_262144_19
void softmax_262144_19() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = me();
  int32_t v3 = v2 * 256;
  int32_t v4 = v1 + v3;
  __dma_aligned float v5[64];
  int32_t v6 = v1 + 4096;
  __dma_aligned float v7[2];
  int32_t v8 = v6 + 64;
  int32_t v9 = v8 + v3;
  __dma_aligned float v10[64];
  mram_read((const __mram_ptr float*)v4, (float*)v5, 64 * sizeof(float));
  mram_read((const __mram_ptr float*)v6, (float*)v7, 2 * sizeof(float));
  for (int32_t v11 = 0; v11 < 64; v11 += 1) {
    float v12 = v5[v11];
    float v13 = v7[0];
    float v14 = v12 / v13;
    v10[v11] = v14;
  }
  mram_write((const float*)v10, (__mram_ptr float*)v9, 64 * sizeof(float));
  return;
}
#endif

#ifdef COMPILE_rmsnorm_262144_opt
void rmsnorm_262144_opt() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = me();
  int32_t v3 = v2 * 256;
  int32_t v4 = v1 + v3;
  __dma_aligned float v5[64];
  int32_t v6 = v1 + 4096;
  int32_t v7 = v6 + v3;
  __dma_aligned float v8[64];
  int32_t v9 = v6 + 4096;
  int32_t v10 = v2 * 8;
  int32_t v11 = v9 + v10;
  __dma_aligned float v12[2];
  mram_read((const __mram_ptr float*)v4, (float*)v5, 64 * sizeof(float));
  mram_read((const __mram_ptr float*)v7, (float*)v8, 64 * sizeof(float));
  v12[0] = (float)0.0e+00;
  v12[1] = (float)0.0e+00;
  for (int32_t v13 = 0; v13 < 64; v13 += 2) {
    int32_t v14 = v13 + 1;
    float v15 = v5[v13];
    float v16 = v8[v13];
    float v17 = v12[0];
    float v18 = v15 * v16;
    float v19 = v18 + v17;
    v12[0] = v19;
    float v20 = v5[v14];
    float v21 = v8[v14];
    float v22 = v12[1];
    float v23 = v20 * v21;
    float v24 = v23 + v22;
    v12[1] = v24;
  }
  mram_write((const float*)v12, (__mram_ptr float*)v11, 2 * sizeof(float));
  return;
}
#endif

#ifdef COMPILE_rmsnorm_262144_opt_20
void rmsnorm_262144_opt_20() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = me();
  int32_t v3 = v2 * 256;
  int32_t v4 = v1 + v3;
  __dma_aligned float v5[64];
  int32_t v6 = v1 + 4096;
  int32_t v7 = v6 + v3;
  __dma_aligned float v8[64];
  int32_t v9 = v6 + 4096;
  __dma_aligned float v10[2];
  int32_t v11 = v9 + 64;
  int32_t v12 = v11 + v3;
  __dma_aligned float v13[64];
  mram_read((const __mram_ptr float*)v4, (float*)v5, 64 * sizeof(float));
  mram_read((const __mram_ptr float*)v7, (float*)v8, 64 * sizeof(float));
  mram_read((const __mram_ptr float*)v9, (float*)v10, 2 * sizeof(float));
  float v14 = v10[0];
  for (int32_t v15 = 0; v15 < 64; v15 += 1) {
    float v16 = v5[v15];
    float v17 = v8[v15];
    float v18 = v16 * v17;
    float v19 = v18 * v14;
    v13[v15] = v19;
  }
  mram_write((const float*)v13, (__mram_ptr float*)v12, 64 * sizeof(float));
  return;
}
#endif

#ifdef COMPILE_softmax_262144_opt
void softmax_262144_opt() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = me();
  int32_t v3 = v2 * 256;
  int32_t v4 = v1 + v3;
  __dma_aligned float v5[64];
  int32_t v6 = v1 + 4096;
  __dma_aligned float v7[2];
  int32_t v8 = v6 + 64;
  int32_t v9 = v8 + v3;
  __dma_aligned float v10[64];
  int32_t v11 = v8 + 4096;
  int32_t v12 = v2 * 8;
  int32_t v13 = v11 + v12;
  __dma_aligned float v14[2];
  mram_read((const __mram_ptr float*)v4, (float*)v5, 64 * sizeof(float));
  mram_read((const __mram_ptr float*)v6, (float*)v7, 2 * sizeof(float));
  float v15 = v7[0];
  for (int32_t v16 = 0; v16 < 64; v16 += 2) {
    int32_t v17 = v16 + 1;
    float v18 = v5[v16];
    float v19 = v14[0];
    float v20 = v18 - v15;
    float v21 = expf(v20);
    float v22 = v19 + v21;
    v10[v16] = v21;
    v14[0] = v22;
    float v23 = v5[v17];
    float v24 = v14[1];
    float v25 = v23 - v15;
    float v26 = expf(v25);
    float v27 = v24 + v26;
    v10[v17] = v21;
    v14[1] = v27;
  }
  mram_write((const float*)v10, (__mram_ptr float*)v9, 64 * sizeof(float));
  mram_write((const float*)v14, (__mram_ptr float*)v13, 2 * sizeof(float));
  return;
}
#endif

#ifdef COMPILE_softmax_262144_opt_21
void softmax_262144_opt_21() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = me();
  int32_t v3 = v2 * 256;
  int32_t v4 = v1 + v3;
  __dma_aligned float v5[64];
  int32_t v6 = v1 + 4096;
  __dma_aligned float v7[2];
  int32_t v8 = v6 + 64;
  int32_t v9 = v8 + v3;
  __dma_aligned float v10[64];
  mram_read((const __mram_ptr float*)v4, (float*)v5, 64 * sizeof(float));
  mram_read((const __mram_ptr float*)v6, (float*)v7, 2 * sizeof(float));
  float v11 = v7[0];
  for (int32_t v12 = 0; v12 < 64; v12 += 1) {
    float v13 = v5[v12];
    float v14 = v13 / v11;
    v10[v12] = v14;
  }
  mram_write((const float*)v10, (__mram_ptr float*)v9, 64 * sizeof(float));
  return;
}
#endif

#ifdef COMPILE_va_262144
void va_262144() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = me();
  int32_t v3 = v2 * 256;
  int32_t v4 = v1 + v3;
  __dma_aligned float v5[64];
  int32_t v6 = v1 + 4096;
  int32_t v7 = v6 + v3;
  __dma_aligned float v8[64];
  int32_t v9 = v6 + 4096;
  int32_t v10 = v9 + v3;
  __dma_aligned float v11[64];
  mram_read((const __mram_ptr float*)v4, (float*)v5, 64 * sizeof(float));
  mram_read((const __mram_ptr float*)v7, (float*)v8, 64 * sizeof(float));
  for (int32_t v12 = 0; v12 < 64; v12 += 1) {
    float v13 = v5[v12];
    float v14 = v8[v12];
    float v15 = v13 + v14;
    v11[v12] = v15;
  }
  mram_write((const float*)v11, (__mram_ptr float*)v10, 64 * sizeof(float));
  return;
}
#endif

#ifdef COMPILE_mha_big
void mha_big() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = me();
  int32_t v3 = v2 * 64;
  int32_t v4 = v1 + v3;
  __dma_aligned float v5[16];
  int32_t v6 = v1 + 1024;
  int32_t v7 = v6 + v3;
  __dma_aligned float v8[16];
  int32_t v9 = v6 + 1024;
  int32_t v10 = v9 + v3;
  __dma_aligned float v11[16];
  mram_read((const __mram_ptr float*)v4, (float*)v5, 16 * sizeof(float));
  mram_read((const __mram_ptr float*)v7, (float*)v8, 16 * sizeof(float));
  for (int32_t v12 = 0; v12 < 16; v12 += 1) {
    float v13 = v5[v12];
    float v14 = v8[v12];
    float v15 = v13 * v14;
    v11[v12] = v15;
  }
  mram_write((const float*)v11, (__mram_ptr float*)v10, 16 * sizeof(float));
  return;
}
#endif

#ifdef COMPILE_mha_big_22
void mha_big_22() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = me();
  int32_t v3 = v2 * 16;
  int32_t v4 = v1 + v3;
  __dma_aligned float v5[4];
  int32_t v6 = v1 + 256;
  __dma_aligned float v7[2];
  int32_t v8 = v6 + 64;
  int32_t v9 = v8 + v3;
  __dma_aligned float v10[4];
  mram_read((const __mram_ptr float*)v4, (float*)v5, 4 * sizeof(float));
  mram_read((const __mram_ptr float*)v6, (float*)v7, 2 * sizeof(float));
  for (int32_t v11 = 0; v11 < 4; v11 += 1) {
    float v12 = v5[v11];
    float v13 = v7[0];
    float v14 = v12 - v13;
    v10[v11] = v14;
  }
  mram_write((const float*)v10, (__mram_ptr float*)v9, 4 * sizeof(float));
  return;
}
#endif

#ifdef COMPILE_mha_big_23
void mha_big_23() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = me();
  int32_t v3 = v2 * 16;
  int32_t v4 = v1 + v3;
  __dma_aligned float v5[4];
  int32_t v6 = v1 + 256;
  __dma_aligned float v7[2];
  int32_t v8 = v6 + 64;
  int32_t v9 = v8 + v3;
  __dma_aligned float v10[4];
  mram_read((const __mram_ptr float*)v4, (float*)v5, 4 * sizeof(float));
  mram_read((const __mram_ptr float*)v6, (float*)v7, 2 * sizeof(float));
  for (int32_t v11 = 0; v11 < 4; v11 += 1) {
    float v12 = v5[v11];
    float v13 = v7[0];
    float v14 = v12 / v13;
    v10[v11] = v14;
  }
  mram_write((const float*)v10, (__mram_ptr float*)v9, 4 * sizeof(float));
  return;
}
#endif

#ifdef COMPILE_mha_big_24
void mha_big_24() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = me();
  int32_t v3 = v2 * 64;
  int32_t v4 = v1 + v3;
  __dma_aligned float v5[16];
  int32_t v6 = v1 + 1024;
  __dma_aligned float v7[2];
  int32_t v8 = v6 + 64;
  int32_t v9 = v8 + v3;
  __dma_aligned float v10[16];
  mram_read((const __mram_ptr float*)v4, (float*)v5, 16 * sizeof(float));
  mram_read((const __mram_ptr float*)v6, (float*)v7, 2 * sizeof(float));
  for (int32_t v11 = 0; v11 < 16; v11 += 1) {
    float v12 = v5[v11];
    float v13 = v7[0];
    float v14 = v12 * v13;
    v10[v11] = v14;
  }
  mram_write((const float*)v10, (__mram_ptr float*)v9, 16 * sizeof(float));
  return;
}
#endif

#ifdef COMPILE_mha_big_25
void mha_big_25() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = me();
  int32_t v3 = v2 * 64;
  int32_t v4 = v1 + v3;
  __dma_aligned float v5[16];
  int32_t v6 = v1 + 1024;
  int32_t v7 = v6 + v3;
  __dma_aligned float v8[16];
  int32_t v9 = v6 + 1024;
  int32_t v10 = v9 + v3;
  __dma_aligned float v11[16];
  mram_read((const __mram_ptr float*)v4, (float*)v5, 16 * sizeof(float));
  mram_read((const __mram_ptr float*)v7, (float*)v8, 16 * sizeof(float));
  for (int32_t v12 = 0; v12 < 16; v12 += 1) {
    float v13 = v5[v12];
    float v14 = v8[v12];
    float v15 = v13 + v14;
    v11[v12] = v15;
  }
  mram_write((const float*)v11, (__mram_ptr float*)v10, 16 * sizeof(float));
  return;
}
#endif

#ifdef COMPILE_test_0
void test_0() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = me();
  int32_t v3 = v2 * 16;
  int32_t v4 = v1 + v3;
  __dma_aligned int32_t v5[4];
  int32_t v6 = v1 + 256;
  int32_t v7 = v6 + v3;
  __dma_aligned int32_t v8[4];
  int32_t v9 = v6 + 256;
  int32_t v10 = v9 + v3;
  __dma_aligned int32_t v11[4];
  mram_read((const __mram_ptr int32_t*)v4, (int32_t*)v5, 4 * sizeof(int32_t));
  mram_read((const __mram_ptr int32_t*)v7, (int32_t*)v8, 4 * sizeof(int32_t));
  for (int32_t v12 = 0; v12 < 4; v12 += 1) {
    int32_t v13 = v5[v12];
    int32_t v14 = v8[v12];
    int32_t v15 = v13 + v14;
    v11[v12] = v15;
  }
  mram_write((const int32_t*)v11, (__mram_ptr int32_t*)v10, 4 * sizeof(int32_t));
  return;
}
#endif

#ifdef COMPILE_test_2
void test_2() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = me();
  int32_t v3 = v2 * 16;
  int32_t v4 = v1 + v3;
  __dma_aligned int32_t v5[4];
  int32_t v6 = v1 + 256;
  int32_t v7 = v6 + v3;
  __dma_aligned int32_t v8[4];
  int32_t v9 = v6 + 256;
  int32_t v10 = v9 + v3;
  __dma_aligned int32_t v11[4];
  int32_t v12 = v9 + 256;
  int32_t v13 = v12 + v3;
  __dma_aligned int32_t v14[4];
  int32_t v15 = v12 + 256;
  int32_t v16 = v15 + v3;
  __dma_aligned int32_t v17[4];
  int32_t v18 = v15 + 256;
  int32_t v19 = v18 + v3;
  __dma_aligned int32_t v20[4];
  int32_t v21 = v18 + 256;
  int32_t v22 = v21 + v3;
  __dma_aligned int32_t v23[4];
  int32_t v24 = v21 + 256;
  int32_t v25 = v24 + v3;
  __dma_aligned int32_t v26[4];
  int32_t v27 = v24 + 256;
  int32_t v28 = v27 + v3;
  __dma_aligned int32_t v29[4];
  mram_read((const __mram_ptr int32_t*)v4, (int32_t*)v5, 4 * sizeof(int32_t));
  mram_read((const __mram_ptr int32_t*)v7, (int32_t*)v8, 4 * sizeof(int32_t));
  mram_read((const __mram_ptr int32_t*)v10, (int32_t*)v11, 4 * sizeof(int32_t));
  mram_read((const __mram_ptr int32_t*)v13, (int32_t*)v14, 4 * sizeof(int32_t));
  mram_read((const __mram_ptr int32_t*)v16, (int32_t*)v17, 4 * sizeof(int32_t));
  mram_read((const __mram_ptr int32_t*)v19, (int32_t*)v20, 4 * sizeof(int32_t));
  for (int32_t v30 = 0; v30 < 4; v30 += 1) {
    int32_t v31 = v5[v30];
    int32_t v32 = v8[v30];
    int32_t v33 = v31 + v32;
    v23[v30] = v33;
  }
  for (int32_t v34 = 0; v34 < 4; v34 += 1) {
    int32_t v35 = v11[v34];
    int32_t v36 = v14[v34];
    int32_t v37 = v35 + v36;
    v26[v34] = v37;
  }
  for (int32_t v38 = 0; v38 < 4; v38 += 1) {
    int32_t v39 = v17[v38];
    int32_t v40 = v20[v38];
    int32_t v41 = v39 + v40;
    v29[v38] = v41;
  }
  mram_write((const int32_t*)v23, (__mram_ptr int32_t*)v22, 4 * sizeof(int32_t));
  mram_write((const int32_t*)v26, (__mram_ptr int32_t*)v25, 4 * sizeof(int32_t));
  mram_write((const int32_t*)v29, (__mram_ptr int32_t*)v28, 4 * sizeof(int32_t));
  return;
}
#endif

BARRIER_INIT(my_barrier, NR_TASKLETS);

int main(void) {
  barrier_wait(&my_barrier);
  mem_reset();
#ifdef COMPILE_forward
  forward();
#endif
#ifdef COMPILE_forward_3
  forward_3();
#endif
#ifdef COMPILE_forward_6
  forward_6();
#endif
#ifdef COMPILE_forward_8
  forward_8();
#endif
#ifdef COMPILE_mha
  mha();
#endif
#ifdef COMPILE_mha_9
  mha_9();
#endif
#ifdef COMPILE_rmsnorm
  rmsnorm();
#endif
#ifdef COMPILE_rmsnorm_11
  rmsnorm_11();
#endif
#ifdef COMPILE_softmax
  softmax();
#endif
#ifdef COMPILE_softmax_13
  softmax_13();
#endif
#ifdef COMPILE_rmsnorm_1048576
  rmsnorm_1048576();
#endif
#ifdef COMPILE_rmsnorm_1048576_14
  rmsnorm_1048576_14();
#endif
#ifdef COMPILE_softmax_1048576
  softmax_1048576();
#endif
#ifdef COMPILE_softmax_1048576_16
  softmax_1048576_16();
#endif
#ifdef COMPILE_va_1048576
  va_1048576();
#endif
#ifdef COMPILE_rmsnorm_262144
  rmsnorm_262144();
#endif
#ifdef COMPILE_rmsnorm_262144_17
  rmsnorm_262144_17();
#endif
#ifdef COMPILE_softmax_262144
  softmax_262144();
#endif
#ifdef COMPILE_softmax_262144_19
  softmax_262144_19();
#endif
#ifdef COMPILE_rmsnorm_262144_opt
  rmsnorm_262144_opt();
#endif
#ifdef COMPILE_rmsnorm_262144_opt_20
  rmsnorm_262144_opt_20();
#endif
#ifdef COMPILE_softmax_262144_opt
  softmax_262144_opt();
#endif
#ifdef COMPILE_softmax_262144_opt_21
  softmax_262144_opt_21();
#endif
#ifdef COMPILE_va_262144
  va_262144();
#endif
#ifdef COMPILE_mha_big
  mha_big();
#endif
#ifdef COMPILE_mha_big_22
  mha_big_22();
#endif
#ifdef COMPILE_mha_big_23
  mha_big_23();
#endif
#ifdef COMPILE_mha_big_24
  mha_big_24();
#endif
#ifdef COMPILE_mha_big_25
  mha_big_25();
#endif
#ifdef COMPILE_test_0
  test_0();
#endif
#ifdef COMPILE_test_2
  test_2();
#endif
  mem_reset();
  return 0;
}
