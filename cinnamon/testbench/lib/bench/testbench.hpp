#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>

// todo call srand(0) before
template <typename T, unsigned... Shape> T *init_matrix() {
  size_t size = 1;
  printf("%ld\n" , size);
  size_t shape[] = {Shape...};
  for (auto dim : shape) {
    size *= dim;
    printf("dim %ld size %ld\n" , dim, size);
  }

  T *mat = (T *)malloc(size * sizeof(T));
  for (size_t i = 0; i < size; i++) {
    while ((mat[i] = (T)(rand() % 100)) == 0)
      ;
  }
  return mat;
}

inline void timeAndExecute(int reps, int warmup, const char *name,
                           std::function<void()> run) {
  using std::chrono::duration;
  using std::chrono::duration_cast;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;

  for (int i = 0; i < warmup; i++) {
    run();
  }
  auto start = high_resolution_clock::now();
  for (int i = 0; i < reps; i++) {
    run();
  }
  auto stop = high_resolution_clock::now();

  duration<double, std::milli> ms_int =
      duration_cast<milliseconds>(start - stop);
  printf("Average time (ms) over %d reps (%s): %f", reps, name,
         ms_int.count() / reps);
}

#define DO_BENCH(reps, warmup, code)                                           \
  do {                                                                         \
    timeAndExecute((reps), (warmup), "" #code, [&]() { code; });               \
  } while (false);