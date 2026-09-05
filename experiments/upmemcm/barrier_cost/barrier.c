
#include <barrier.h>
#include <defs.h>
#include <stdint.h>

#define NITER 100
#define WARM_ITERS 60

BARRIER_INIT(my_barrier, NR_TASKLETS);

int buf[24];

int main(void) {
  // int tid = me();

  // Pre-work: fixed arithmetic loop, result stored to prevent elimination.
  // int x = 1;
  // for (int i = 0; i < WARM_ITERS; i++)
  //   x = x * 2 + 1;
  // buf[tid] = x;

  for (int i = 0; i < NITER; i++) {
    //   buf[tid] += 1;
#ifdef BENCH
    barrier_wait(&my_barrier);
#endif
  }

  // Post-work: same fixed loop reading back from buf.
  // x = buf[tid];
  // for (int i = 0; i < WARM_ITERS; i++)
  //   x = x * 2 + 1;
  // buf[tid] = x;

  return 0;
}
