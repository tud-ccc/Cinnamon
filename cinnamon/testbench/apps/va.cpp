#include "../lib/bench/testbench.hpp"
#include <cstdint>

extern "C" {
void va_8(int32_t *, int32_t *);
void va_16(int32_t *, int32_t *);
}

#define BENCH_VA(ty, M, N, fun_name)                                           \
  do {                                                                         \
    ty *A = init_matrix<ty, M, N>();                                           \
    ty *B = init_matrix<ty, M, N>();                                           \
    DO_BENCH(REPS, WARMUP, fun_name(A, B));                                    \
    free(A);                                                                   \
    free(B);                                                                   \
  } while (false)

int main(void) {
  srand(0);

  BENCH_VA(int32_t, 8, 2097152, va_8);
  BENCH_VA(int32_t, 16, 1048576, va_16);

  return 0;
}