// (r2, r3) = ((X * W1) * W2, X * W3)   (a two-gemm chain plus one gemm
// parallel to it, sharing X); X: 8×N, W1, W2, W3: N×N. Size class = bytes
// of each N×N i32 weight (see 2mm_seq.cpp). Two out-params in result order,
// one backing vector (see 2mm_par.cpp). References wrap in u32 like the
// device's i32 (see matmul_ref_wrap).

#include "common.hpp"

extern "C" void BENCH_FN(DTY *, DTY *, DTY *, DTY *, DTY *, DTY *);

struct Mm3ParSeq {
  static constexpr bench::Size kSizes[] = {
      {"_3mm_1MB", {512}},
      {"_3mm_16MB", {2048}},
      {"_3mm_64MB", {4096}},
      {"_3mm_256MB", {8192}},
  };

  static constexpr size_t d = 8;
  size_t n;
  std::vector<DTY> X, W1, W2, W3, out; // out = [r2: d*n | r3: d*n]

  void setup(const size_t *dims) {
    n = dims[0];
    X = bench::random_vector(d * n);
    W1 = bench::random_vector(n * n);
    W2 = bench::random_vector(n * n);
    W3 = bench::random_vector(n * n);
    out = bench::output_vector(2 * d * n);
    printf("%s  N=%zu", TOSTR(BENCH_FN), n);
  }

  void run() {
    BENCH_FN(X.data(), W1.data(), W2.data(), W3.data(), out.data(),
             out.data() + d * n);
  }
  const std::vector<DTY> &output() const { return out; }

  std::vector<double> reference() const {
    std::vector<uint32_t> t(d * n), r(2 * d * n);
    bench::matmul_ref_wrap(X.data(), W1.data(), d, n, n, t.data());
    bench::matmul_ref_wrap(t.data(), W2.data(), d, n, n, r.data());
    bench::matmul_ref_wrap(X.data(), W3.data(), d, n, n, r.data() + d * n);
    return bench::wrapped_golden(r);
  }
};

int main(int argc, char **argv) { return bench::run<Mm3ParSeq>(argc, argv); }
