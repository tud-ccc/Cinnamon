// Bench driver for mmtv (prim_mmtv) functions.
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

extern "C" void BENCH_FN(DTY *, DTY *, DTY *);

extern "C" {
void upmemrt_start_stat_collection(int iter);
void upmemrt_dump_stats(const char *prefix);
}

struct MmtvSpec {
  size_t b, m, k;
};

static MmtvSpec spec() {
  const char *fn = TOSTR(BENCH_FN);
  if (!strcmp(fn, "mmtv_4MB"))
    return {32, 64, 512};
  if (!strcmp(fn, "mmtv_64MB"))
    return {128, 256, 512};
  if (!strcmp(fn, "mmtv_256MB"))
    return {256, 512, 512};
  if (!strcmp(fn, "mmtv_512MB"))
    return {512, 512, 512};
  fprintf(stderr, "mmtv.cpp: unknown function '%s' — add it to spec()\n", fn);
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

  MmtvSpec s = spec();
  srand(0);
  DTY *A = alloc_mat(s.b * s.m * s.k);
  DTY *x = alloc_mat(s.b * s.k);
  DTY *out = alloc_mat(s.b * s.m); // output buffer; reused across iters

#if FUNCTIONAL_CHECK
  // todo
  DTY *C = alloc_mat(s.b * s.m);
  for (int b = 0; b < s.b; b++)
    for (int m = 0; m < s.m; m++)
      C[b * s.m + m] = 0;
  for (int k = 0; k < s.k; k++)
    C[b * s.m + m] += A[b * s.m * s.k + m * s.k + k] * x[b * s.k + k];
#endif

  printf("%s  M=%zu N=%zu  iters=%d\n", TOSTR(BENCH_FN), s.m, s.k, iters);
  fflush(stdout);

  uint64_t *elapsed_ns = (uint64_t *)malloc(iters * sizeof(uint64_t));
  assert(elapsed_ns);

  for (int iter = 0; iter < iters; iter++) {
    upmemrt_start_stat_collection(iter);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    BENCH_FN(A, x, out);
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
