#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// Prototypes for the four generated host functions.
// Each takes a bare pointer (tensor<N xi32> → int32_t *) and returns
// the reduction result as a scalar.
extern "C" {
  int32_t red_4MB(int32_t *);
  int32_t red_64MB(int32_t *);
  int32_t red_256MB(int32_t *);
  int32_t red_512MB(int32_t *);
}

// Forward-declare the stats API without pulling in dpu.h.
extern "C" {
  void upmemrt_start_stat_collection(int iter);
  void upmemrt_dump_stats(const char *prefix);
}

static const int  ITERS      = 5;
static const char OUT_DIR[]  = "output";

// Allocate and fill a buffer of N int32 elements with random non-zero values.
template <size_t N>
static int32_t *alloc_input() {
  auto *buf = (int32_t *)malloc(N * sizeof(int32_t));
  assert(buf);
  for (size_t i = 0; i < N; i++)
    buf[i] = (int32_t)(rand() % 100 + 1);
  return buf;
}

// Run fn(buf) ITERS times, calling the stats API around each iteration, then
// dump CSV files to OUT_DIR/<name>_{scatter,gather,launch}.csv.
template <size_t N>
static void bench_one(const char *name, int32_t (*fn)(int32_t *)) {
  int32_t *buf = alloc_input<N>();

  printf("=== %s  (%zu elements, %zu MB) ===\n",
         name, N, N * sizeof(int32_t) / (1024 * 1024));
  fflush(stdout);

  for (int iter = 0; iter < ITERS; iter++) {
    upmemrt_start_stat_collection(iter);
    volatile int32_t result = fn(buf);
    (void)result;
    printf("  iter %d done\n", iter);
    fflush(stdout);
  }

  char prefix[512];
  snprintf(prefix, sizeof(prefix), "%s/%s", OUT_DIR, name);
  upmemrt_dump_stats(prefix);

  free(buf);
}

int main() {
  srand(0);

  bench_one<524288>  ("red_4MB",   red_4MB);
  bench_one<8388608> ("red_64MB",  red_64MB);
  bench_one<34554432>("red_256MB", red_256MB);
  bench_one<67108864>("red_512MB", red_512MB);

  return 0;
}
