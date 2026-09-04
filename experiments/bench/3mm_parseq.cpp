// (r2, r3) = ((X * W1) * W2, X * W3)   (a two-gemm chain plus one gemm
// parallel to it, sharing X); X: 8xK, W1, W3: KxM, W2: MxK. Size class =
// bytes of each i32 weight (see 2mm_seq.cpp). Two out-params in result
// order, one backing vector (see 2mm_par.cpp). References wrap in u32 like
// the device's i32 (see matmul_ref_wrap).

#include "common.hpp"

extern "C" void BENCH_FN(DTY *, DTY *, DTY *, DTY *, DTY *, DTY *);

struct Mm3ParSeq {
  static constexpr bench::Size kSizes[] = {
      {"_3mm_1MB", {512, 512}},
      {"_3mm_64MB", {4096, 4096}},
      {"_3mm_256MB", {8192, 8192}},
      {"_3mm_512MB", {8192, 16384}},
  };

  static constexpr size_t d = 8;
  size_t k, m;
  std::vector<DTY> X, W1, W2, W3, out; // out = [r2: d*k | r3: d*m]

  void setup(const size_t *dims) {
    k = dims[0];
    m = dims[1];
    X = bench::random_vector(d * k);
    W1 = bench::random_vector(k * m);
    W2 = bench::random_vector(m * k);
    W3 = bench::random_vector(k * m);
    out = bench::output_vector(d * k + d * m);
    printf("%s  K=%zu M=%zu", TOSTR(BENCH_FN), k, m);
  }

  void run() {
    BENCH_FN(X.data(), W1.data(), W2.data(), W3.data(), out.data(),
             out.data() + d * k);
  }
  const std::vector<DTY> &output() const { return out; }

  std::vector<double> reference() const {
    std::vector<uint32_t> t(d * m), r(d * k + d * m);
    bench::matmul_ref_wrap(X.data(), W1.data(), d, k, m, t.data());
    bench::matmul_ref_wrap(t.data(), W2.data(), d, m, k, r.data());
    bench::matmul_ref_wrap(X.data(), W3.data(), d, k, m, r.data() + d * k);
    return bench::wrapped_golden(r);
  }
};

int main(int argc, char **argv) { return bench::run<Mm3ParSeq>(argc, argv); }
