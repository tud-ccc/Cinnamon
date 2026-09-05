// DPU kernel ATiM's UPMEM backend generates for va_1048576_1_1.reproduced.tir.py.
// Regenerate with `python points/dump_atim_c.py --only va_1048576_1_1.reproduced`.
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
__mram_noinit int32_t A[512];
__mram_noinit int32_t B[512];
__mram int32_t C[512];
int main() {
  const int blockIdx_x = blockIdx.x;
  const int blockIdx_y = blockIdx.y;
  const int blockIdx_z = blockIdx.z;

  unsigned int tasklet_id = me();
  if (tasklet_id == 0) mem_reset();
  barrier_wait(&barrier);
  int32_t* A_local = (int32_t*) mem_alloc(128 * sizeof(int32_t));
  int32_t* B_local = (int32_t*) mem_alloc(128 * sizeof(int32_t));
  int32_t* C_local = (int32_t*) mem_alloc(2 * sizeof(int32_t));
  mram_read((__mram_ptr void*)(A + (tasklet_id * 128)), A_local + 0, 512);
  mram_read((__mram_ptr void*)(B + (tasklet_id * 128)), B_local + 0, 512);
  for (int32_t i_3 = 0; i_3 < 64; ++i_3) {
    for (int32_t i_4 = 0; i_4 < 2; ++i_4) {
      int32_t cse_var_1 = ((i_3 * 2) + i_4);
      C_local[i_4] = (A_local[cse_var_1] + B_local[cse_var_1]);
    }
    mram_write(C_local + 0, (__mram_ptr void*)(C + ((tasklet_id * 128) + (i_3 * 2))), 8);
  }
}
