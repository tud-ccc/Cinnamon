
#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <perfcounter.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

BARRIER_INIT(my_barrier, NR_TASKLETS);

int fib(int n) {
  int a = 1, b = 1; 
  for (int i = 0; i < n; i++) {
    int b0 = b;
    b = a + b;
    a = b0;
  }
  return b;
}

int[24] buf;

int main(void) {
  int tid = me();
  buf[tid] = fib(tid);
  for (int i = 0; i < 100; i++) {
    buf[tid] *= i;
#ifdef BENCH
    barrier_wait(&my_barrier);
#endif
  }
  buf[tid] += fib(tid);
  return 0;
}
