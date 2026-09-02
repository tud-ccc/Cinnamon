// (r2, r3) = ((A * B) * C, A * D)   (a two-gemm chain plus one gemm parallel
// to it, sharing A); A: d×128, B: 128×256, C: 256×128, D: 128×128. Two
// out-params in result order, one backing vector (see 2mm_par.cpp). Inputs
// are 0/1 so the chained stage stays exact in i32.

#include "common.hpp"

extern "C" void BENCH_FN(DTY *, DTY *, DTY *, DTY *, DTY *, DTY *);

struct Mm3ParSeq {
  static constexpr bench::Size kSizes[] = {
      {"_3mm_d8", {8}},
      {"_3mm_d16", {16}},
      {"_3mm_d32", {32}},
      {"_3mm_d64", {64}},
  };

  size_t d;
  std::vector<DTY> A, B, C, D, out; // out = [r2: d*128 | r3: d*128]

  void setup(const size_t *dims) {
    d = dims[0];
    A = bench::random_vector(d * 128, 2);
    B = bench::random_vector(128 * 256, 2);
    C = bench::random_vector(256 * 128, 2);
    D = bench::random_vector(128 * 128, 2);
    out = bench::output_vector(d * 128 + d * 128);
    printf("%s  d=%zu", TOSTR(BENCH_FN), d);
  }

  void run() {
    BENCH_FN(A.data(), B.data(), C.data(), D.data(), out.data(),
             out.data() + d * 128);
  }
  const std::vector<DTY> &output() const { return out; }

  std::vector<double> reference() const {
    std::vector<double> t(d * 256), r(d * 128 + d * 128);
    bench::matmul_ref(A.data(), B.data(), d, 128, 256, t.data());
    bench::matmul_ref(t.data(), C.data(), d, 256, 128, r.data());
    bench::matmul_ref(A.data(), D.data(), d, 128, 128, r.data() + d * 128);
    return r;
  }
};

int main(int argc, char **argv) { return bench::run<Mm3ParSeq>(argc, argv); }
