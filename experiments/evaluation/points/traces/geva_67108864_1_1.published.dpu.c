// DPU kernel ATiM's UPMEM backend generates for geva_67108864_1_1.published.tir.py.
// Regenerate with `python points/dump_atim_c.py --only geva_67108864_1_1.published`.
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
__mram_noinit int32_t B[32768];
__mram int32_t C[32768];
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
  int32_t* A_local = (int32_t*) mem_alloc(16 * sizeof(int32_t));
  int32_t* B_local = (int32_t*) mem_alloc(16 * sizeof(int32_t));
  for (int32_t i_2 = 0; i_2 < 256; ++i_2) {
    mram_read((__mram_ptr void*)(A + ((tasklet_id * 4096) + (i_2 * 16))), A_local + 0, 64);
    mram_read((__mram_ptr void*)(B + ((tasklet_id * 4096) + (i_2 * 16))), B_local + 0, 64);
    #pragma clang loop unroll(full)
    for (int32_t i_3 = 0; i_3 < 4; ++i_3) {
      #pragma clang loop unroll(full)
      for (int32_t i_4 = 0; i_4 < 4; ++i_4) {
        int32_t cse_var_1 = ((i_3 * 4) + i_4);
        A_local[cse_var_1] = ((alpha_val * A_local[cse_var_1]) + (beta_val * B_local[cse_var_1]));
      }
    }
    mram_write(A_local + 0, (__mram_ptr void*)(C + ((tasklet_id * 4096) + (i_2 * 16))), 64);
  }
}
