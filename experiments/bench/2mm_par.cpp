// (r, r2) = (A * B, A * C)   (two independent cinm.op.gemm sharing A);
// A: d×128, B: 128×256, C: 128×128. The two results come back through two
// out-params (buffer-results-to-out-params appends them in result order);
// one backing vector holds both segments so output()/reference() compare
// them in one pass. Inputs are 0/1 like the other 2mm/3mm drivers.

#include "common.hpp"

extern "C" void BENCH_FN(DTY *, DTY *, DTY *, DTY *, DTY *);

struct Mm2Par {
  static constexpr bench::Size kSizes[] = {
      {"_2mm_par_d8", {8}},
      {"_2mm_par_d16", {16}},
      {"_2mm_par_d32", {32}},
      {"_2mm_par_d64", {64}},
  };

  size_t d;
  std::vector<DTY> A, B, C, out; // out = [r: d*256 | r2: d*128]

  void setup(const size_t *dims) {
    d = dims[0];
    A = bench::random_vector(d * 128, 2);
    B = bench::random_vector(128 * 256, 2);
    C = bench::random_vector(128 * 128, 2);
    out = bench::output_vector(d * 256 + d * 128);
    printf("%s  d=%zu", TOSTR(BENCH_FN), d);
  }

  void run() {
    BENCH_FN(A.data(), B.data(), C.data(), out.data(), out.data() + d * 256);
  }
  const std::vector<DTY> &output() const { return out; }

  std::vector<double> reference() const {
    std::vector<double> r(d * 256 + d * 128);
    bench::matmul_ref(A.data(), B.data(), d, 128, 256, r.data());
    bench::matmul_ref(A.data(), C.data(), d, 128, 128, r.data() + d * 256);
    return r;
  }
};

int main(int argc, char **argv) { return bench::run<Mm2Par>(argc, argv); }
