#include "dpu_lib.h"
void run_va_kernel() {
  init_tasklet();
  int32_t v1 = me();
  int32_t v2 = 0;
  int32_t v3 = 1;
  int32_t v4 = 128;
  int32_t v5 = 64;
  int32_t v6 = v4 * v5;
  int32_t v7 = (uint32_t)DPU_MRAM_HEAP_POINTER;
  int32_t v8 = v5 * v1;
  int32_t v9 = v8 + v7;
  int32_t v10 = v7 + v6;
  int32_t v11 = v10 + v8;
  int * v12 = (int *)dpu_wram_alloc (64*sizeof(int));
  int * v13 = (int *)dpu_wram_alloc (64*sizeof(int));
  int32_t v14;
  int32_t v15;
  int32_t v16 = v9;
  int32_t v17 = v11;
  for (int32_t v18 = v2; v18 < v4; v18 += v3) {
    dpu_memcpy (MRAMToWRAM,v12, v5*sizeof(int), v16*sizeof(int));
    dpu_memcpy (MRAMToWRAM,v13, v5*sizeof(int), v17*sizeof(int));
    for (int32_t v19 = v2; v19 < v5; v19 += v3) {
      int32_t v20 = v12[v19];
      int32_t v21 = v13[v19];
      int32_t v22 = v20 + v21;
      v12[v19] = v22;
    };
    dpu_memcpy (WRAMToMRAM,v12, v5*sizeof(int), v16*sizeof(int));
    int32_t v23 = v16 + v5;
    int32_t v24 = v17 + v5;
    v16 = v23;
    v17 = v24;
  }
  v14 = v16;
  v15 = v17;
  return;
}