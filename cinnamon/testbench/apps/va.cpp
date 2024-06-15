#include "../lib/bench/testbench.hpp"


// linked with LLVM module
extern "C" {
	void va_8(int32_t*, int32_t*);
	void va_16(int32_t*, int32_t*);
}

int main(void) {
  srand(0);

  unsigned int reps = 3;
  unsigned int warmup = 3;

  int32_t *A = init_matrix<int32_t, 8, 2097152>();
  int32_t *B = init_matrix<int32_t, 8, 2097152>();
  
  DO_BENCH(reps, warmup, va_8(A, B));
  DO_BENCH(reps, warmup, va_16(A, B));

  free(A);
  free(B);

  return 0;
}