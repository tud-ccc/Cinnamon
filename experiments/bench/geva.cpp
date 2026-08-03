// Bench driver for gemv (prim_gemv) functions.
//
// Requires only -DBENCH_FN=<fn_name>; dimensions are looked up at runtime.
//
// Function signature produced by cinm lowering (bare-ptr calling convention,
// return tensor becomes an extra out-param):
//   extern "C" void BENCH_FN(float *A, float *x, float *out);
// where A is M×N (row-major), x is N, out is M.
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

extern "C" void BENCH_FN(DTY *, DTY *, DTY, DTY, DTY *);

extern "C" {
void upmemrt_start_stat_collection(int iter);
void upmemrt_dump_stats(const char *prefix);
}

static size_t M() {
  const char *fn = TOSTR(BENCH_FN);
  if (!strcmp(fn, "va_4MB"))
    return 1048576;
  if (!strcmp(fn, "va_64MB"))
    return 16777216;
  if (!strcmp(fn, "va_256MB"))
    return 67108864;
  fprintf(stderr, "mtv.cpp: unknown function '%s' — add it to spec()\n", fn);
  exit(1);
}

static DTY *alloc_mat(size_t n) {
  DTY *p = (DTY *)malloc(n * sizeof(DTY));
  assert(p && "malloc failed");
  for (size_t i = 0; i < n; i++)
    p[i] = (DTY)(rand() % 1000 + 1) / 100.0f;
  return p;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <output_dir> [<iters>]\n", argv[0]);
    return 1;
  }
  const char *out_dir = argv[1];
  int iters = argc > 2 ? atoi(argv[2]) : 5;

  size_t m = M();
  srand(0);
  DTY *A = alloc_mat(m);
  DTY *x = alloc_mat(m);
  DTY c = rand() % 2046;
  DTY d = rand() % 2476;
  DTY *out = alloc_mat(m); // output buffer; reused across iters

  printf("%s  M=%zu iters=%d\n", TOSTR(BENCH_FN), m, iters);
  fflush(stdout);

  uint64_t *elapsed_ns = (uint64_t *)malloc(iters * sizeof(uint64_t));
  assert(elapsed_ns);

  for (int iter = 0; iter < iters; iter++) {
    upmemrt_start_stat_collection(iter);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    BENCH_FN(A, x, c, d, out);
    clock_gettime(CLOCK_MONOTONIC, &t1);
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
  free(out);
  free(x);
  free(A);
  return 0;
}
