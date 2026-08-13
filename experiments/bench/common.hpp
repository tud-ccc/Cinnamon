
#include <cassert>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <memory>
#include <vector>

#ifndef BENCH_FN
#error "BENCH_FN must be defined at compile time (-DBENCH_FN=<function_name>)"
#endif

#ifndef DTY
#define DTY int32_t
#endif
#define STRINGIFY(x) #x
#define TOSTR(x) STRINGIFY(x)

static DTY my_rand() { return (DTY)(rand() % 128); }

static std::vector<DTY> alloc_mat(size_t n) {
  std::vector<DTY> p(n);
  for (size_t i = 0; i < n; i++)
    p[i] = my_rand();
  return p;
}

extern "C" {
void upmemrt_start_stat_collection(int iter);
void upmemrt_dump_stats(const char *prefix);
}
