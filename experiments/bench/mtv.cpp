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

#include "common.hpp"

extern "C" void BENCH_FN(DTY *, DTY *, DTY *);

struct GemvSpec {
  size_t m, n;
};

static GemvSpec spec() {
  const char *fn = TOSTR(BENCH_FN);
  if (!strcmp(fn, "mtv_4MB"))
    return {1024, 1024};
  if (!strcmp(fn, "mtv_64MB"))
    return {4096, 4096};
  if (!strcmp(fn, "mtv_256MB"))
    return {8192, 8192};
  if (!strcmp(fn, "mtv_512MB"))
    return {8192, 16384};
  fprintf(stderr, "mtv.cpp: unknown function '%s' — add it to spec()\n", fn);
  exit(1);
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <output_dir> [<iters>]\n", argv[0]);
    return 1;
  }
  const char *out_dir = argv[1];
  int iters = argc > 2 ? atoi(argv[2]) : 5;

  GemvSpec s = spec();
  srand(0);
  auto A = alloc_mat(s.m * s.n);
  auto x = alloc_mat(s.n);
  auto out = alloc_mat(s.m); // output buffer; reused across iters

  printf("%s  M=%zu N=%zu  iters=%d\n", TOSTR(BENCH_FN), s.m, s.n, iters);
  fflush(stdout);

  std::vector<uint64_t> elapsed_ns(iters);

  for (int iter = 0; iter < iters; iter++) {
    upmemrt_start_stat_collection(iter);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    BENCH_FN(A.data(), x.data(), out.data());
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

  return 0;
}
