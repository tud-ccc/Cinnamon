// DPU kernel ATiM's UPMEM backend generates for gemv_8192_1_8192.published.tir.py.
// Regenerate with `python points/dump_atim_c.py --only gemv_8192_1_8192.published`.
// Function: main_kernel
#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <barrier.h>
#include <seqread.h>
#include <handshake.h>

typedef struct { int32_t x, y, z; } BlockInfo;
BARRIER_INIT(barrier, NR_TASKLETS);

__host BlockInfo blockIdx;

inline int min(int x, int y) { return x < y ? x : y; }
inline int max(int x, int y) { return x > y ? x : y; }
__mram_noinit int32_t A[32768];
__mram int32_t B[128];
__mram int32_t C_rf_global[256];
__host int32_t alpha_val_[2];
int main() {
  const int blockIdx_x = blockIdx.x;
  const int blockIdx_y = blockIdx.y;
  const int blockIdx_z = blockIdx.z;

  const int32_t alpha_val = alpha_val_[0];
  unsigned int tasklet_id = me();
  if (tasklet_id == 0) mem_reset();
  barrier_wait(&barrier);
  int32_t* C_rf_global_local = (int32_t*) mem_alloc(8 * sizeof(int32_t));
  int32_t* A_local = (int32_t*) mem_alloc(256 * sizeof(int32_t));
  int32_t* B_local = (int32_t*) mem_alloc(32 * sizeof(int32_t));
  for (int32_t i_2 = 0; i_2 < 2; ++i_2) {
    #pragma clang loop unroll(full)
    for (int32_t i_3_init = 0; i_3_init < 8; ++i_3_init) {
      C_rf_global_local[i_3_init] = 0;
    }
    for (int32_t k_1_0 = 0; k_1_0 < 4; ++k_1_0) {
      #pragma clang loop unroll(full)
      for (int32_t ax0 = 0; ax0 < 8; ++ax0) {
        int32_t cse_var_1 = (ax0 * 32);
        mram_read((__mram_ptr void*)(A + ((((tasklet_id * 2048) + (i_2 * 1024)) + (k_1_0 * 256)) + cse_var_1)), A_local + cse_var_1, 128);
      }
      mram_read((__mram_ptr void*)(B + (k_1_0 * 32)), B_local + 0, 128);
      #pragma clang loop unroll(full)
      for (int32_t i_3 = 0; i_3 < 8; ++i_3) {
        #pragma clang loop unroll(full)
        for (int32_t k_1_1 = 0; k_1_1 < 32; ++k_1_1) {
          C_rf_global_local[i_3] = (C_rf_global_local[i_3] + ((alpha_val * A_local[((i_3 * 32) + k_1_1)]) * B_local[k_1_1]));
        }
      }
    }
    mram_write(C_rf_global_local + 0, (__mram_ptr void*)(C_rf_global + ((tasklet_id * 16) + (i_2 * 8))), 32);
  }
}
