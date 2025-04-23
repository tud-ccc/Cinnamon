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

#define BENCH_MM(ty, M, K, N, fun_name)                                        \
  do {                                                                         \
    ty *A = init_matrix<ty, M, K>();                                           \
    ty *B = init_matrix<ty, K, N>();                                           \
    DO_BENCH(REPS, WARMUP, fun_name(A, B));                                    \
    free(A);                                                                   \
    free(B);                                                                   \
  } while (false)

int main(void) {
  srand(0);

  BENCH_MM(int32_t, 25088, 16, 256, mm_dimm4_nopt);
  BENCH_MM(int32_t, 6272, 64, 256, mm_dimm4_opt);

  BENCH_MM(int32_t, 25088, 8, 256, mm_dimm8_nopt);
  BENCH_MM(int32_t, 3136, 64, 256, mm_dimm8_opt);

  BENCH_MM(int32_t, 25088, 4, 256, mm_dimm16_nopt);
  BENCH_MM(int32_t, 1568, 64, 256, mm_dimm16_opt);

  return 0;
}
