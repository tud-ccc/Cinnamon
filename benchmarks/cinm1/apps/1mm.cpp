// One gemm per DIMM count. See apps/README.md for how this suite's drivers
// differ from the prim and multiop ones.
#include "../../common.hpp"

#include <cstdint>

extern "C" {
void mm_dimm4_nopt(int32_t *, int32_t *, int32_t *);
void mm_dimm4_opt(int32_t *, int32_t *, int32_t *);
void mm_dimm8_nopt(int32_t *, int32_t *, int32_t *);
void mm_dimm8_opt(int32_t *, int32_t *, int32_t *);
void mm_dimm16_nopt(int32_t *, int32_t *, int32_t *);
void mm_dimm16_opt(int32_t *, int32_t *, int32_t *);
}

namespace {

/// One m×k · k×n kernel: fill, time, and verify against a wrapped reference.
/// The reference wraps because these kernels accumulate in i32 and the larger
/// shapes overflow it by construction (see matmul_ref_wrap).
bool bench_mm(const char *name, size_t m, size_t k, size_t n,
              void (*fn)(int32_t *, int32_t *, int32_t *)) {
  auto a = bench::random_vector(m * k);
  auto b = bench::random_vector(k * n);
  auto out = bench::output_vector(m * n);

  double ms = bench::time_mean_ms(bench::kReps, bench::kWarmup,
                                  [&] { fn(a.data(), b.data(), out.data()); });
  printf("%-16s %10.3f ms\n", name, ms);

  std::vector<uint32_t> want(m * n);
  bench::matmul_ref_wrap(a.data(), b.data(), m, k, n, want.data());
  return bench::check(out, bench::wrapped_golden(want), name);
}

} // namespace

int main() {
  srand(0);
  bool ok = true;
  ok &= bench_mm("mm_dimm4_nopt", 8, 1024, 256, mm_dimm4_nopt);
  ok &= bench_mm("mm_dimm4_opt", 16, 1024, 128, mm_dimm4_opt);
  ok &= bench_mm("mm_dimm8_nopt", 8, 1024, 128, mm_dimm8_nopt);
  ok &= bench_mm("mm_dimm8_opt", 16, 1024, 64, mm_dimm8_opt);
  ok &= bench_mm("mm_dimm16_nopt", 8, 1024, 512, mm_dimm16_nopt);
  ok &= bench_mm("mm_dimm16_opt", 16, 1024, 512, mm_dimm16_opt);
  return ok ? 0 : 1;
}
