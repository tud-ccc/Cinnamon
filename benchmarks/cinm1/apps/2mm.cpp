// Two chained gemms, (A*B)*C, one kernel per DIMM count. See apps/README.md
// for how this suite's drivers differ from the prim and multiop ones.
#include "../../common.hpp"

#include <cstdint>

extern "C" {
void mm_dimm4_nopt(int32_t *, int32_t *, int32_t *, int32_t *);
void mm_dimm4_opt(int32_t *, int32_t *, int32_t *, int32_t *);
void mm_dimm8_nopt(int32_t *, int32_t *, int32_t *, int32_t *);
void mm_dimm8_opt(int32_t *, int32_t *, int32_t *, int32_t *);
}

namespace {

/// (m×k · k×n) · n×p. The intermediate stays in uint32_t so the second stage
/// consumes exactly the wrapped i32 the first one produced -- these shapes
/// overflow i32 by construction, and that wraparound is part of what is
/// verified (see matmul_ref_wrap).
bool bench_2mm(const char *name, size_t m, size_t k, size_t n, size_t p,
               void (*fn)(int32_t *, int32_t *, int32_t *, int32_t *)) {
  auto a = bench::random_vector(m * k);
  auto b = bench::random_vector(k * n);
  auto c = bench::random_vector(n * p);
  auto out = bench::output_vector(m * p);

  double ms = bench::time_mean_ms(bench::kReps, bench::kWarmup, [&] {
    fn(a.data(), b.data(), c.data(), out.data());
  });
  printf("%-16s %10.3f ms\n", name, ms);

  std::vector<uint32_t> ab(m * n);
  bench::matmul_ref_wrap(a.data(), b.data(), m, k, n, ab.data());
  std::vector<uint32_t> want(m * p);
  bench::matmul_ref_wrap(ab.data(), c.data(), m, n, p, want.data());
  return bench::check(out, bench::wrapped_golden(want), name);
}

} // namespace

int main() {
  srand(0);
  bool ok = true;
  ok &= bench_2mm("mm_dimm4_nopt", 8, 1024, 256, 2048, mm_dimm4_nopt);
  ok &= bench_2mm("mm_dimm4_opt", 16, 1024, 128, 2048, mm_dimm4_opt);
  ok &= bench_2mm("mm_dimm8_nopt", 8, 1024, 128, 2048, mm_dimm8_nopt);
  ok &= bench_2mm("mm_dimm8_opt", 8, 1024, 128, 2048, mm_dimm8_opt);
  return ok ? 0 : 1;
}
