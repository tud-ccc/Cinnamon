// Elementwise add, one kernel per DIMM count. See apps/README.md for how this
// suite's drivers differ from the prim and multiop ones.
#include "../../common.hpp"

#include <cstdint>

extern "C" {
void va_8(int32_t *, int32_t *, int32_t *);
void va_16(int32_t *, int32_t *, int32_t *);
}

namespace {

bool bench_va(const char *name, size_t m, size_t n,
              void (*fn)(int32_t *, int32_t *, int32_t *)) {
  const size_t total = m * n;
  auto a = bench::random_vector(total);
  auto b = bench::random_vector(total);
  auto out = bench::output_vector(total);

  double ms = bench::time_mean_ms(bench::kReps, bench::kWarmup,
                                  [&] { fn(a.data(), b.data(), out.data()); });
  printf("%-16s %10.3f ms\n", name, ms);

  return bench::check(out, bench::axpby_ref(a, b, 1.0, 1.0, total), name);
}

} // namespace

int main() {
  srand(0);
  bool ok = true;
  ok &= bench_va("va_8", 8, 2097152, va_8);
  ok &= bench_va("va_16", 16, 1048576, va_16);
  return ok ? 0 : 1;
}
