// Bench driver for reduce (prim_red) functions.
//
// Requires only -DBENCH_FN=<fn_name>; problem size is looked up from the
// function name at runtime so the compilation Makefile needs no BENCH_N.
//
// Function signature produced by cinm lowering (bare-ptr calling convention):
//   extern "C" int32_t BENCH_FN(int32_t *input);
//
// Binary interface: bench_<fn> <output_dir> [<iters>]

#include <cassert>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#ifndef BENCH_FN
#error "BENCH_FN must be defined at compile time (-DBENCH_FN=<function_name>)"
#endif

#define DTY int32_t
#define STRINGIFY(x) #x
#define TOSTR(x) STRINGIFY(x)

extern "C" DTY BENCH_FN(DTY *);

extern "C" {
void upmemrt_start_stat_collection(int iter);
void upmemrt_dump_stats(const char *prefix);
}

static size_t n_elements() {
  const char *fn = TOSTR(BENCH_FN);
  if (!strcmp(fn, "red_4MB"))
    return 524288;
  if (!strcmp(fn, "red_64MB"))
    return 8388608;
  if (!strcmp(fn, "red_256MB"))
    return 34554432;
  if (!strcmp(fn, "red_512MB"))
    return 67108864;
  fprintf(stderr, "red.cpp: unknown function '%s' — add it to n_elements()\n",
          fn);
  exit(1);
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <output_dir> [<iters>]\n", argv[0]);
    return 1;
  }
  const char *out_dir = argv[1];
  int iters = argc > 2 ? atoi(argv[2]) : 5;

  size_t n = n_elements();
  DTY *buf = (DTY *)malloc(n * sizeof(DTY));
  assert(buf && "malloc failed");
  srand(0);
  for (size_t i = 0; i < n; i++)
    buf[i] = (DTY)(rand() % 100 + 1);

  printf("%s  n=%zu  iters=%d\n", TOSTR(BENCH_FN), n, iters);
  fflush(stdout);

  uint64_t *elapsed_ns = (uint64_t *)malloc(iters * sizeof(uint64_t));
  assert(elapsed_ns);

  for (int iter = 0; iter < iters; iter++) {
    upmemrt_start_stat_collection(iter);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    volatile DTY result = BENCH_FN(buf);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    (void)result;
    elapsed_ns[iter] = (uint64_t)(t1.tv_sec - t0.tv_sec) * 1000000000ULL +
                       (uint64_t)(t1.tv_nsec - t0.tv_nsec);
    printf("  iter %d  %.3f ms\n", iter, elapsed_ns[iter] / 1e6);
    fflush(stdout);
  }

  char prefix[1024];
  snprintf(prefix, sizeof(prefix), "%s/" TOSTR(BENCH_FN), out_dir);
  upmemrt_dump_stats(prefix);

  char total_path[1024];
  snprintf(total_path, sizeof(total_path), "%s/" TOSTR(BENCH_FN) "_total.csv",
           out_dir);
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
