// DPU kernel ATiM's UPMEM backend generates for red_524288_1_1.reproduced.tir.py.
// Regenerate with `python points/dump_atim_c.py --only red_524288_1_1.reproduced`.
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
int64_t B_rf_global_rf_shared[8];
int64_t red_buf0[8];
__mram_noinit int64_t A[4096];
__mram int64_t B_rf_global[1];
int main() {
  const int blockIdx_x = blockIdx.x;
  const int blockIdx_y = blockIdx.y;
  const int blockIdx_z = blockIdx.z;

  unsigned int tasklet_id = me();
  if (tasklet_id == 0) mem_reset();
  barrier_wait(&barrier);
  int64_t* B_rf_global_rf_shared_local = (int64_t*) mem_alloc(1 * sizeof(int64_t));
    int64_t* A_local = (int64_t*) mem_alloc(256 * sizeof(int64_t));
    B_rf_global_rf_shared_local[0] = (int64_t)0;
  for (int32_t i_1_1_0 = 0; i_1_1_0 < 2; ++i_1_1_0) {
    mram_read((__mram_ptr void*)(A + ((tasklet_id * 512) + (i_1_1_0 * 256))), A_local + 0, 2048);
    for (int32_t i_1_1_1 = 0; i_1_1_1 < 32; ++i_1_1_1) {
      for (int32_t i_1_1_2 = 0; i_1_1_2 < 8; ++i_1_1_2) {
        B_rf_global_rf_shared_local[0] = (B_rf_global_rf_shared_local[0] + A_local[((i_1_1_1 * 8) + i_1_1_2)]);
      }
    }
    B_rf_global_rf_shared[tasklet_id] = B_rf_global_rf_shared_local[0];
  }
  barrier_wait(&barrier);
  red_buf0[tasklet_id] = B_rf_global_rf_shared[tasklet_id];
  barrier_wait(&barrier);
  if (tasklet_id < 4) {
    handshake_wait_for((tasklet_id + 4));
    red_buf0[tasklet_id] = (red_buf0[tasklet_id] + red_buf0[(tasklet_id + 4)]);
  }
  if (4 <= tasklet_id) {
    handshake_notify();
  }
  if (tasklet_id < 2) {
    handshake_wait_for((tasklet_id + 2));
    red_buf0[tasklet_id] = (red_buf0[tasklet_id] + red_buf0[(tasklet_id + 2)]);
  }
  if ((2 <= tasklet_id) && (tasklet_id < 4)) {
    handshake_notify();
  }
  if (tasklet_id < 1) {
    handshake_wait_for((tasklet_id + 1));
    red_buf0[tasklet_id] = (red_buf0[tasklet_id] + red_buf0[(tasklet_id + 1)]);
  }
  if (tasklet_id == 1) {
    handshake_notify();
  }
  if (tasklet_id == 0) {
    B_rf_global[0] = red_buf0[0];
  }
}
