#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <ctime>

// BENCH_FN and BENCH_N are set by the compiler (e.g. -DBENCH_FN=red_4MB -DBENCH_N=524288).
#ifndef BENCH_FN
#error "BENCH_FN must be defined at compile time (-DBENCH_FN=<function_name>)"
#endif
#ifndef BENCH_N
#error "BENCH_N must be defined at compile time (-DBENCH_N=<num_elements>)"
#endif

#define STRINGIFY(x) #x
#define TOSTR(x)     STRINGIFY(x)

extern "C" int32_t BENCH_FN(int32_t *);

extern "C" {
  void upmemrt_start_stat_collection(int iter);
  void upmemrt_dump_stats(const char *prefix);
}

static constexpr size_t N = (size_t)BENCH_N;

// argv: bench_<fn> <output_dir> [<iters>]
int main(int argc, char *argv[]) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <output_dir> [<iters>]\n", argv[0]);
    return 1;
  }

  const char *out  = argv[1];
  int iters        = argc > 2 ? atoi(argv[2]) : 5;

  int32_t *buf = (int32_t *)malloc(N * sizeof(int32_t));
  assert(buf && "malloc failed");
  srand(0);
  for (size_t i = 0; i < N; i++)
    buf[i] = (int32_t)(rand() % 100 + 1);

  printf("%s  N=%zu  iters=%d\n", TOSTR(BENCH_FN), N, iters);
  fflush(stdout);

  uint64_t *elapsed_ns = (uint64_t *)malloc(iters * sizeof(uint64_t));
  assert(elapsed_ns);

  for (int iter = 0; iter < iters; iter++) {
    upmemrt_start_stat_collection(iter);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    volatile int32_t result = BENCH_FN(buf);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    (void)result;
    elapsed_ns[iter] = (uint64_t)(t1.tv_sec - t0.tv_sec) * 1000000000ULL
                       + (uint64_t)(t1.tv_nsec - t0.tv_nsec);
    printf("  iter %d  %.3f ms\n", iter, elapsed_ns[iter] / 1e6);
    fflush(stdout);
  }

  char prefix[1024];
  snprintf(prefix, sizeof(prefix), "%s/" TOSTR(BENCH_FN), out);
  upmemrt_dump_stats(prefix);

  // Dump total wall-clock timings.
  char total_path[1024];
  snprintf(total_path, sizeof(total_path), "%s/" TOSTR(BENCH_FN) "_total.csv", out);
  FILE *f = fopen(total_path, "w");
  assert(f && "failed to open total CSV");
  fprintf(f, "iter,elapsed_ns\n");
  for (int i = 0; i < iters; i++)
    fprintf(f, "%d,%" PRIu64 "\n", i, elapsed_ns[i]);
  fclose(f);
  free(elapsed_ns);

  free(buf);
  return 0;
}
