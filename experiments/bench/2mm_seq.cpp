// r2 = (X * W1) * W2   (two chained cinm.op.gemm); X: 8×N, W1, W2: N×N.
// The size class names the bytes of each N×N i32 weight (1MB -> N=512,
// 16MB -> 2048, 64MB -> 4096, 256MB -> 8192). The reference wraps in u32
// like the device's i32 (see matmul_ref_wrap), so operands use the default
// range.

#include "common.hpp"

extern "C" void BENCH_FN(DTY *, DTY *, DTY *, DTY *);

struct Mm2Seq {
  static constexpr bench::Size kSizes[] = {
      {"_2mm_seq_1MB", {512}},
      {"_2mm_seq_16MB", {2048}},
      {"_2mm_seq_64MB", {4096}},
      {"_2mm_seq_256MB", {8192}},
  };

  static constexpr size_t d = 8;
  size_t n;
  std::vector<DTY> X, W1, W2, out;

  void setup(const size_t *dims) {
    n = dims[0];
    X = bench::random_vector(d * n);
    W1 = bench::random_vector(n * n);
    W2 = bench::random_vector(n * n);
    out = bench::output_vector(d * n);
    printf("%s  N=%zu", TOSTR(BENCH_FN), n);
  }

  void run() { BENCH_FN(X.data(), W1.data(), W2.data(), out.data()); }
  const std::vector<DTY> &output() const { return out; }

  std::vector<double> reference() const {
    std::vector<uint32_t> t(d * n), r(d * n);
    bench::matmul_ref_wrap(X.data(), W1.data(), d, n, n, t.data());
    bench::matmul_ref_wrap(t.data(), W2.data(), d, n, n, r.data());
    return bench::wrapped_golden(r);
  }
};

int main(int argc, char **argv) { return bench::run<Mm2Seq>(argc, argv); }
