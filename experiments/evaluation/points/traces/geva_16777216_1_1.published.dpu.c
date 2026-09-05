// DPU kernel ATiM's UPMEM backend generates for geva_16777216_1_1.published.tir.py.
// Regenerate with `python points/dump_atim_c.py --only geva_16777216_1_1.published`.
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
__mram_noinit int32_t A[8192];
__mram_noinit int32_t B[8192];
__mram int32_t C[8192];
__host int32_t alpha_val_[2];
__host int32_t beta_val_[2];
int main() {
  const int blockIdx_x = blockIdx.x;
  const int blockIdx_y = blockIdx.y;
  const int blockIdx_z = blockIdx.z;

  const int32_t alpha_val = alpha_val_[0];
  const int32_t beta_val = beta_val_[0];
  unsigned int tasklet_id = me();
  if (tasklet_id == 0) mem_reset();
  barrier_wait(&barrier);
  int32_t* C_local = (int32_t*) mem_alloc(512 * sizeof(int32_t));
  int32_t* A_local = (int32_t*) mem_alloc(32 * sizeof(int32_t));
  int32_t* B_local = (int32_t*) mem_alloc(32 * sizeof(int32_t));
  for (int32_t i_2 = 0; i_2 < 2; ++i_2) {
    for (int32_t i_3 = 0; i_3 < 16; ++i_3) {
      mram_read((__mram_ptr void*)(A + (((tasklet_id * 1024) + (i_2 * 512)) + (i_3 * 32))), A_local + 0, 128);
      mram_read((__mram_ptr void*)(B + (((tasklet_id * 1024) + (i_2 * 512)) + (i_3 * 32))), B_local + 0, 128);
      #pragma clang loop unroll(full)
      for (int32_t i_4 = 0; i_4 < 32; ++i_4) {
        C_local[((i_3 * 32) + i_4)] = ((alpha_val * A_local[i_4]) + (beta_val * B_local[i_4]));
      }
    }
    mram_write(C_local + 0, (__mram_ptr void*)(C + ((tasklet_id * 1024) + (i_2 * 512))), 2048);
  }
}
