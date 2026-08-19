// DPU kernel ATiM's UPMEM backend generates for ttv_256_512_512.published.tir.py.
// Regenerate with `python points/dump_atim_c.py --only ttv_256_512_512.published`.
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
__mram int32_t B[512];
__mram int32_t C[64];
int main() {
  const int blockIdx_x = blockIdx.x;
  const int blockIdx_y = blockIdx.y;
  const int blockIdx_z = blockIdx.z;

  unsigned int tasklet_id = me();
  if (tasklet_id == 0) mem_reset();
  barrier_wait(&barrier);
  int32_t* C_local = (int32_t*) mem_alloc(4 * sizeof(int32_t));
  int32_t* A_local = (int32_t*) mem_alloc(64 * sizeof(int32_t));
  int32_t* B_local = (int32_t*) mem_alloc(16 * sizeof(int32_t));
  #pragma clang loop unroll(full)
  for (int32_t j_3_init = 0; j_3_init < 4; ++j_3_init) {
    C_local[j_3_init] = 0;
  }
  for (int32_t k_0 = 0; k_0 < 32; ++k_0) {
    #pragma clang loop unroll(full)
    for (int32_t ax1 = 0; ax1 < 4; ++ax1) {
      int32_t cse_var_1 = (ax1 * 16);
      mram_read((__mram_ptr void*)(A + (((tasklet_id * 2048) + (k_0 * 64)) + cse_var_1)), A_local + cse_var_1, 64);
    }
    mram_read((__mram_ptr void*)(B + (k_0 * 16)), B_local + 0, 64);
    #pragma clang loop unroll(full)
    for (int32_t j_3 = 0; j_3 < 4; ++j_3) {
      #pragma clang loop unroll(full)
      for (int32_t k_1 = 0; k_1 < 8; ++k_1) {
        #pragma clang loop unroll(full)
        for (int32_t k_2 = 0; k_2 < 2; ++k_2) {
          int32_t cse_var_2 = (k_1 * 2);
          C_local[j_3] = (C_local[j_3] + (A_local[(((j_3 * 16) + cse_var_2) + k_2)] * B_local[(cse_var_2 + k_2)]));
        }
      }
      C[((tasklet_id * 4) + j_3)] = C_local[j_3];
    }
  }
}
