
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <dpu.h>

#define DPU_ASSERT(x)                                                          \
  do {                                                                         \
    dpu_error_t _e = (x);                                                      \
    if (_e != DPU_OK) {                                                        \
      fprintf(stderr, "DPU error %d at %s:%d\n", _e, __FILE__, __LINE__);     \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static const int WARMUP = 2;
static const int N = 100;
// Number of barrier iterations per DPU launch (matches barrier.c)
static const int BARRIER_ITERS = 200;

// Returns average nanoseconds per launch over N launches.
static double measure(const char *path) {
  struct dpu_set_t set;
  DPU_ASSERT(dpu_alloc(1, NULL, &set));
  DPU_ASSERT(dpu_load(set, path, NULL));

  for (int i = 0; i < WARMUP; i++)
    DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < N; i++)
    DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));
  auto t1 = std::chrono::steady_clock::now();

  DPU_ASSERT(dpu_free(set));

  double ns = static_cast<double>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
  return ns / N;
}

int main(int argc, char **argv) {
  std::vector<double> barrier_ns(25, 0.0);
  std::vector<double> control_ns(25, 0.0);

  for (int T = 1; T <= 24; T++) {
    char path[64];
    snprintf(path, sizeof(path), "./barrier_%d", T);
    barrier_ns[T] = measure(path);

    snprintf(path, sizeof(path), "./barrier_%d_control", T);
    control_ns[T] = measure(path);
  }

  printf("tasklets,barrier_ns_per_launch,control_ns_per_launch,overhead_ns_per_barrier\n");
  for (int T = 1; T <= 24; T++) {
    double overhead = (barrier_ns[T] - control_ns[T]) / BARRIER_ITERS;
    printf("%d,%.1f,%.1f,%.2f\n", T, barrier_ns[T], control_ns[T], overhead);
  }

  return 0;
}
