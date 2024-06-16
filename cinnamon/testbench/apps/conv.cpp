#include "../lib/bench/testbench.hpp"
#include <cstdint>

// linked with LLVM module
extern "C" {
int32_t *mm_dimm4_nopt(int32_t *, int32_t *);
int32_t *mm_dimm4_opt(int32_t *, int32_t *);
int32_t *mm_dimm8_nopt(int32_t *, int32_t *);
int32_t *mm_dimm8_opt(int32_t *, int32_t *);
int32_t *mm_dimm16_nopt(int32_t *, int32_t *);
int32_t *mm_dimm16_opt(int32_t *, int32_t *);
}

int main(void) {
  srand(0);

  unsigned int reps = 3;
  unsigned int warmup = 3;

  {
    int32_t *A = init_matrix<int32_t, 25088, 16>();
    int32_t *B = init_matrix<int32_t, 16, 256>();

    DO_BENCH(reps, warmup, mm_dimm4_nopt(A, B));
    DO_BENCH(reps, warmup, mm_dimm4_opt(A, B));

    free(A);
    free(B);
  }
  {
    int32_t *A = init_matrix<int32_t, 25088, 8>();
    int32_t *B = init_matrix<int32_t, 8, 256>();

    DO_BENCH(reps, warmup, mm_dimm8_nopt(A, B));
    DO_BENCH(reps, warmup, mm_dimm8_opt(A, B));

    free(A);
    free(B);
  }
  {
    int32_t *A = init_matrix<int32_t, 25088, 4>();
    int32_t *B = init_matrix<int32_t, 4, 256>();

    DO_BENCH(reps, warmup, mm_dimm16_nopt(A, B));
    DO_BENCH(reps, warmup, mm_dimm16_opt(A, B));

    free(A);
    free(B);
  }
  return 0;
}