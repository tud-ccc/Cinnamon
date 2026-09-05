// (r1, r2) = (X * W1, X * W2)   (two independent cinm.op.gemm sharing X --
// the QKV pattern); X: 8xK, W1, W2: KxM. Size class = bytes of each i32
// weight (see 2mm_seq.cpp). The compiled function has two results, so the
// driver passes two out-pointers, in result order, into one backing vector;
// the reference lays its parts out the same way. References wrap in u32
// like the device's i32 (see matmul_ref_wrap).

#include "common.hpp"

extern "C" void BENCH_FN(DTY *, DTY *, DTY *, DTY *, DTY *);

struct Mm2Par {
  static constexpr bench::Size kSizes[] = {
      {"_2mm_par_1MB", {512, 512}},
      {"_2mm_par_64MB", {4096, 4096}},
      {"_2mm_par_256MB", {8192, 8192}},
      {"_2mm_par_512MB", {8192, 16384}},
  };

  static constexpr size_t d = 8;
  size_t k, m;
  std::vector<DTY> X, W1, W2, out; // out = [r1: d*m | r2: d*m]

  void setup(const size_t *dims) {
    k = dims[0];
    m = dims[1];
    X = bench::random_vector(d * k);
    W1 = bench::random_vector(k * m);
    W2 = bench::random_vector(k * m);
    out = bench::output_vector(2 * d * m);
    printf("%s  K=%zu M=%zu", TOSTR(BENCH_FN), k, m);
  }

  void run() {
    BENCH_FN(X.data(), W1.data(), W2.data(), out.data(), out.data() + d * m);
  }
  const std::vector<DTY> &output() const { return out; }

  std::vector<double> reference() const {
    std::vector<uint32_t> r(2 * d * m);
    bench::matmul_ref_wrap(X.data(), W1.data(), d, k, m, r.data());
    bench::matmul_ref_wrap(X.data(), W2.data(), d, k, m, r.data() + d * m);
    return bench::wrapped_golden(r);
  }
};

int main(int argc, char **argv) { return bench::run<Mm2Par>(argc, argv); }
