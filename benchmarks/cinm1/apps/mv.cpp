// Matrix-vector product, one kernel per DIMM count. See apps/README.md for how
// this suite's drivers differ from the prim and multiop ones.
#include "../../common.hpp"

#include <cstdint>

extern "C" {
void mv_dimm4_opt(int32_t *, int32_t *, int32_t *);
void mv_dimm8_nopt(int32_t *, int32_t *, int32_t *);
void mv_dimm16_opt(int32_t *, int32_t *, int32_t *);
}

namespace {

bool bench_mv(const char *name, size_t m, size_t n,
              void (*fn)(int32_t *, int32_t *, int32_t *)) {
  auto a = bench::random_vector(m * n);
  auto x = bench::random_vector(n);
  auto out = bench::output_vector(m);

  double ms = bench::time_mean_ms(bench::kReps, bench::kWarmup,
                                  [&] { fn(a.data(), x.data(), out.data()); });
  printf("%-16s %10.3f ms\n", name, ms);

  return bench::check(out, bench::gemv_ref(a, x, m, n), name);
}

} // namespace

int main() {
  srand(0);
  bool ok = true;
  ok &= bench_mv("mv_dimm4_opt", 4096, 2048, mv_dimm4_opt);
  ok &= bench_mv("mv_dimm8_nopt", 16384, 512, mv_dimm8_nopt);
  ok &= bench_mv("mv_dimm16_opt", 16384, 128, mv_dimm16_opt);
  return ok ? 0 : 1;
}
