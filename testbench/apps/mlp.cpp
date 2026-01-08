#include "../lib/bench/testbench.hpp"
#include <cstdint>

// linked with LLVM module
extern "C" {
int32_t *mm_dimm1_nopt(int32_t *, int32_t *);
int32_t *mm_dimm1_opt(int32_t *, int32_t *);
int32_t *mm_dimm2_nopt(int32_t *, int32_t *);
int32_t *mm_dimm2_opt(int32_t *, int32_t *);
int32_t *mm_dimm4_nopt(int32_t *, int32_t *);
int32_t *mm_dimm4_opt(int32_t *, int32_t *);
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

  BENCH_MM(int32_t, 1, 1024, 512, mm_dimm1_nopt);
  BENCH_MM(int32_t, 16, 64, 512, mm_dimm1_opt);

  BENCH_MM(int32_t, 1, 1024, 256, mm_dimm2_nopt);
  BENCH_MM(int32_t, 16, 64, 256, mm_dimm2_opt);

  //BENCH_MM(int32_t, 1, 1024, 128, mm_dimm4_nopt);
  BENCH_MM(int32_t, 16, 64, 128, mm_dimm4_opt);

  return 0;
}
