
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" {
#include <dpu.h>
}

#define DPU_ASSERT(x)                                                          \
  do {                                                                         \
    dpu_error_t _e = (x);                                                      \
    if (_e != DPU_OK) {                                                        \
      fprintf(stderr, "DPU error %d at %s:%d\n", _e, __FILE__, __LINE__);      \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static const int WARMUP = 2;
static const int N = 100;
// Number of barrier iterations per DPU launch (matches barrier.c)
static const int BARRIER_ITERS = 100;

// Returns nanoseconds per launch for each of the N trials.
static std::vector<double> measure(const char *path) {
  struct dpu_set_t set;
  DPU_ASSERT(dpu_alloc(1, NULL, &set));
  DPU_ASSERT(dpu_load(set, path, NULL));

  for (int i = 0; i < WARMUP; i++)
    DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

  std::vector<double> results(N);
  for (int i = 0; i < N; i++) {
    auto t0 = std::chrono::steady_clock::now();
    DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));
    auto t1 = std::chrono::steady_clock::now();
    results[i] = static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
  }

  DPU_ASSERT(dpu_free(set));
  return results;
}

int main(int argc, char **argv) {
  std::vector<std::vector<double>> barrier_ns(25);
  std::vector<std::vector<double>> control_ns(25);

  for (int T = 1; T <= 24; T++) {
    char path[64];
    snprintf(path, sizeof(path), "./bin/barrier_%d", T);
    barrier_ns[T] = measure(path);

    snprintf(path, sizeof(path), "./bin/barrier_%d_control", T);
    control_ns[T] = measure(path);
  }

  printf("tasklets,trial,barrier_ns,control_ns,overhead_ns_per_barrier\n");
  for (int T = 1; T <= 24; T++) {
    for (int i = 0; i < N; i++) {
      double overhead = (barrier_ns[T][i] - control_ns[T][i]) / BARRIER_ITERS;
      printf("%d,%d,%.1f,%.1f,%.2f\n", T, i, barrier_ns[T][i], control_ns[T][i],
             overhead);
    }
  }

  return 0;
}
