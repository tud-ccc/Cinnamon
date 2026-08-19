// DPU kernel ATiM's UPMEM backend generates for gemv_1024_1_1024.reproduced.tir.py.
// Regenerate with `python points/dump_atim_c.py --only gemv_1024_1_1024.reproduced`.
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
__mram_noinit int32_t A[4096];
__mram int32_t B[64];
__mram int32_t C_rf_global[64];
__host int32_t alpha_val_[2];
int main() {
  const int blockIdx_x = blockIdx.x;
  const int blockIdx_y = blockIdx.y;
  const int blockIdx_z = blockIdx.z;

  const int32_t alpha_val = alpha_val_[0];
  unsigned int tasklet_id = me();
  if (tasklet_id == 0) mem_reset();
  barrier_wait(&barrier);
  int32_t* C_rf_global_local = (int32_t*) mem_alloc(2 * sizeof(int32_t));
  int32_t* A_local = (int32_t*) mem_alloc(16 * sizeof(int32_t));
  int32_t* B_local = (int32_t*) mem_alloc(8 * sizeof(int32_t));
  for (int32_t i_2 = 0; i_2 < 2; ++i_2) {
    #pragma clang loop unroll(full)
    for (int32_t i_4_init = 0; i_4_init < 2; ++i_4_init) {
      C_rf_global_local[i_4_init] = 0;
    }
    for (int32_t k_1_0 = 0; k_1_0 < 8; ++k_1_0) {
      #pragma clang loop unroll(full)
      for (int32_t ax0 = 0; ax0 < 2; ++ax0) {
        int32_t cse_var_1 = (ax0 * 8);
        mram_read((__mram_ptr void*)(A + ((((tasklet_id * 256) + (i_2 * 128)) + (k_1_0 * 16)) + cse_var_1)), A_local + cse_var_1, 32);
      }
      mram_read((__mram_ptr void*)(B + (k_1_0 * 8)), B_local + 0, 32);
      #pragma clang loop unroll(full)
      for (int32_t k_1_1 = 0; k_1_1 < 8; ++k_1_1) {
        #pragma clang loop unroll(full)
        for (int32_t i_4 = 0; i_4 < 2; ++i_4) {
          C_rf_global_local[i_4] = (C_rf_global_local[i_4] + ((alpha_val * A_local[((i_4 * 8) + k_1_1)]) * B_local[k_1_1]));
        }
      }
      mram_write(C_rf_global_local + 0, (__mram_ptr void*)(C_rf_global + ((tasklet_id * 4) + (i_2 * 2))), 8);
    }
  }
}
