// DPU kernel ATiM's UPMEM backend generates for mmtv_256_512_512.published.tir.py.
// Regenerate with `python points/dump_atim_c.py --only mmtv_256_512_512.published`.
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
  int32_t* C_local = (int32_t*) mem_alloc(1 * sizeof(int32_t));
  int32_t* A_local = (int32_t*) mem_alloc(64 * sizeof(int32_t));
  int32_t* B_local = (int32_t*) mem_alloc(64 * sizeof(int32_t));
  for (int32_t j_2 = 0; j_2 < 4; ++j_2) {
    C_local[0] = 0;
    for (int32_t k_0 = 0; k_0 < 8; ++k_0) {
      int32_t cse_var_1 = (k_0 * 64);
      mram_read((__mram_ptr void*)(A + (((tasklet_id * 2048) + (j_2 * 512)) + cse_var_1)), A_local + 0, 256);
      mram_read((__mram_ptr void*)(B + cse_var_1), B_local + 0, 256);
      for (int32_t k_1 = 0; k_1 < 8; ++k_1) {
        for (int32_t k_2 = 0; k_2 < 8; ++k_2) {
          int32_t cse_var_2 = ((k_1 * 8) + k_2);
          C_local[0] = (C_local[0] + (A_local[cse_var_2] * B_local[cse_var_2]));
        }
      }
    }
    C[((tasklet_id * 4) + j_2)] = C_local[0];
  }
}
