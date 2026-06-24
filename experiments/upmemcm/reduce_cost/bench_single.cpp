#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// BENCH_FN is set by the compiler via -DBENCH_FN=red_4MB etc.
#ifndef BENCH_FN
#error "BENCH_FN must be defined at compile time (-DBENCH_FN=<function_name>)"
#endif

#define STRINGIFY(x) #x
#define TOSTR(x)     STRINGIFY(x)

extern "C" int32_t BENCH_FN(int32_t *);

extern "C" {
  void upmemrt_start_stat_collection(int iter);
  void upmemrt_dump_stats(const char *prefix);
}

// argv: bench_<fn> <num_elements> <output_dir> [<iters>]
int main(int argc, char *argv[]) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <num_elements> <output_dir> [<iters>]\n", argv[0]);
    return 1;
  }

  size_t N         = (size_t)atoll(argv[1]);
  const char *out  = argv[2];
  int iters        = argc > 3 ? atoi(argv[3]) : 5;

  int32_t *buf = (int32_t *)malloc(N * sizeof(int32_t));
  assert(buf && "malloc failed");
  srand(0);
  for (size_t i = 0; i < N; i++)
    buf[i] = (int32_t)(rand() % 100 + 1);

  printf("%s  N=%zu  iters=%d\n", TOSTR(BENCH_FN), N, iters);
  fflush(stdout);

  for (int iter = 0; iter < iters; iter++) {
    upmemrt_start_stat_collection(iter);
    volatile int32_t result = BENCH_FN(buf);
    (void)result;
    printf("  iter %d done\n", iter);
    fflush(stdout);
  }

  char prefix[1024];
  snprintf(prefix, sizeof(prefix), "%s/" TOSTR(BENCH_FN), out);
  upmemrt_dump_stats(prefix);

  free(buf);
  return 0;
}
