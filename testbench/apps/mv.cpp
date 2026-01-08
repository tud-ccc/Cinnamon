#include "../lib/bench/testbench.hpp"
#include <cstdint>

extern "C" {
// int32_t *mv_dimm4_nopt(int32_t *, int32_t *);
int32_t *mv_dimm4_opt(int32_t *, int32_t *);
int32_t *mv_dimm8_nopt(int32_t *, int32_t *);
// int32_t* mv_dimm8_opt(int32_t *, int32_t *);
// int32_t* mv_dimm16_nopt(int32_t *, int32_t *);
int32_t *mv_dimm16_opt(int32_t *, int32_t *);
}

#define BENCH_MV(ty, M, N, fun_name)                                           \
  do {                                                                         \
    ty *A = init_matrix<ty, M, N>();                                           \
    ty *B = init_matrix<ty, N>();                                              \
    DO_BENCH(REPS, WARMUP, fun_name(A, B));                                    \
    free(A);                                                                   \
    free(B);                                                                   \
  } while (false)

int main(void) {
  srand(0);

  //  BENCH_MV(int32_t, 128, 65536, mv_dimm4_nopt);
  BENCH_MV(int32_t, 4096, 2048, mv_dimm4_opt);

  BENCH_MV(int32_t, 16384, 512, mv_dimm8_nopt);
  // BENCH_MV(int32_t, 4096, 2048, mv_dimm8_opt);

  // BENCH_MV(int32_t, 128, 65536, mv_dimm16_nopt);
  BENCH_MV(int32_t, 16384, 128, mv_dimm16_opt);

  return 0;
}