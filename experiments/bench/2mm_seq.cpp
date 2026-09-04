// r2 = (X * W1) * W2   (two chained cinm.op.gemm); X: 8xK, W1: KxM,
// W2: MxK. The size class names the bytes of each i32 weight, both of which
// hold K*M elements: 1MB (512x512), 64MB (4096x4096), 256MB (8192x8192),
// 512MB (8192x16384 -- no square i32 matrix is 512MB, and this is the shape
// the prim suite's 512MB gemv uses). The reference wraps in u32 like the
// device's i32 (see matmul_ref_wrap), so operands use the default range.

#include "common.hpp"

extern "C" void BENCH_FN(DTY *, DTY *, DTY *, DTY *);

struct Mm2Seq {
  static constexpr bench::Size kSizes[] = {
      {"_2mm_seq_1MB", {512, 512}},
      {"_2mm_seq_64MB", {4096, 4096}},
      {"_2mm_seq_256MB", {8192, 8192}},
      {"_2mm_seq_512MB", {8192, 16384}},
  };

  static constexpr size_t d = 8;
  size_t k, m;
  std::vector<DTY> X, W1, W2, out;

  void setup(const size_t *dims) {
    k = dims[0];
    m = dims[1];
    X = bench::random_vector(d * k);
    W1 = bench::random_vector(k * m);
    W2 = bench::random_vector(m * k);
    out = bench::output_vector(d * k);
    printf("%s  K=%zu M=%zu", TOSTR(BENCH_FN), k, m);
  }

  void run() { BENCH_FN(X.data(), W1.data(), W2.data(), out.data()); }
  const std::vector<DTY> &output() const { return out; }

  std::vector<double> reference() const {
    std::vector<uint32_t> t(d * m), r(d * k);
    bench::matmul_ref_wrap(X.data(), W1.data(), d, k, m, t.data());
    bench::matmul_ref_wrap(t.data(), W2.data(), d, m, k, r.data());
    return bench::wrapped_golden(r);
  }
};

int main(int argc, char **argv) { return bench::run<Mm2Seq>(argc, argv); }
