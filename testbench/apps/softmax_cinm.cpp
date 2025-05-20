#include "../lib/bench/testbench.hpp"
#include <cstdint>

extern "C" {
void *softmax_1048576(float *, float *);
}

#define BENCH_SM(ty, M, fun_name)                                              \
  do {                                                                         \
    ty *in = init_matrix<ty, M>();                                             \
    ty *out = init_matrix<ty, M>();                                            \
    DO_BENCH(REPS, WARMUP, fun_name(in, out));                                 \
    free(in);                                                                  \
    free(out);                                                                 \
  } while (false)

int main(void) {
  srand(0);

  BENCH_SM(float, 1048576, softmax_1048576);

  return 0;
}