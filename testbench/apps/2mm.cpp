#include "../lib/bench/testbench.hpp"
#include <cstdint>

extern "C" {
void *mm_dimm4_nopt(int32_t *, int32_t *, int32_t *, int32_t *);
void *mm_dimm4_opt(int32_t *, int32_t *, int32_t *, int32_t *);
void *mm_dimm8_nopt(int32_t *, int32_t *, int32_t *, int32_t *);
void *mm_dimm8_opt(int32_t *, int32_t *, int32_t *, int32_t *);
void *mm_dimm16_nopt(int32_t *, int32_t *, int32_t *, int32_t *);
void *mm_dimm16_opt(int32_t *, int32_t *, int32_t *, int32_t *);
}

#define BENCH_MM(ty, M, K, K2, N, fun_name)                                    \
  do {                                                                         \
    ty *A = init_matrix<ty, M, K>();                                           \
    ty *B = init_matrix<ty, K, K2>();                                          \
    ty *C = init_matrix<ty, K2, N>();                                          \
    ty *OUT = init_matrix<ty, M, N>();                                         \
    DO_BENCH(REPS, WARMUP, fun_name(A, B, C, OUT));                            \
    free(A);                                                                   \
    free(B);                                                                   \
    free(C);                                                                   \
    free(OUT);                                                                 \
  } while (false)

int main(void) {
  srand(0);
  /*

    func.func @mm_dimm4_nopt(%A: tensor<8x1024xi32>, %B: tensor<1024x256xi32>,
    %C: tensor<256x2048xi32>) -> tensor<8x2048xi32> {

    func.func @mm_dimm4_opt(%A: tensor<16x1024xi32>, %B: tensor<1024x128xi32>,
    %C: tensor<128x2048xi32>) -> tensor<16x2048xi32> {

    func.func @mm_dimm8_nopt(%A: tensor<8x1024xi32>, %B: tensor<1024x128xi32>,
    %C: tensor<128x2048xi32>) -> tensor<8x2048xi32>

    */

  BENCH_MM(int32_t, 8, 1024, 256, 2048, mm_dimm4_nopt);
  BENCH_MM(int32_t, 16, 1024, 128, 2048, mm_dimm4_opt);

  BENCH_MM(int32_t, 8, 1024, 128, 2048, mm_dimm8_nopt);
  // BENCH_MM(int32_t, 16, 1024, 64, mm_dimm8_opt);

  // BENCH_MM(int32_t, 8, 1024, 64, mm_dimm16_nopt);
  // BENCH_MM(int32_t, 16, 1024, 32, mm_dimm16_opt);

  return 0;
}