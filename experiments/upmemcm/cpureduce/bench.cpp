#include <cassert>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#ifndef M_VAL
#error "M_VAL must be defined at compile time (-DM_VAL=<num_elements>)"
#endif

extern "C" int32_t bench(int32_t *, int32_t *);

static constexpr size_t M = (size_t)M_VAL;

static constexpr int WARMUP = 5;
static constexpr int BENCH_REPS = 200;

// argv[1]: path to CSV file to append results to (default: output/results.csv)
int main(int argc, char *argv[]) {
  const char *csv_path = argc > 1 ? argv[1] : "output/results.csv";

  int32_t *A = (int32_t *)aligned_alloc(64, M * sizeof(int32_t));
  int32_t *B = (int32_t *)aligned_alloc(64, M * sizeof(int32_t));
  assert(A && B && "allocation failed");

  srand(42);
  for (size_t i = 0; i < M; i++)
    A[i] = (int32_t)rand();

  for (int i = 0; i < WARMUP; i++) {
    volatile int32_t r = bench(A, B);
    (void)r;
  }

  FILE *f = fopen(csv_path, "a");
  assert(f && "failed to open CSV");

  for (int rep = 0; rep < BENCH_REPS; rep++) {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    volatile int32_t r = bench(A, B);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    (void)r;

    uint64_t elapsed_ns = (uint64_t)(t1.tv_sec - t0.tv_sec) * 1000000000ULL +
                          (uint64_t)(t1.tv_nsec - t0.tv_nsec);
    // time_iter: wall time divided by number of loop iterations in bench()
    double time_iter_ns = (double)elapsed_ns / M;
    fprintf(f, "%zu,%.3f\n", M, time_iter_ns);
  }

  fclose(f);
  free(A);
  free(B);
  return 0;
}
