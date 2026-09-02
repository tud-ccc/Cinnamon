// j = (A * B) * (C * D)   (diamond: two independent gemms join in a third);
// A: d×1024, B: 1024×256, C: 256×512, D: 512×2048. Inputs are 0/1 so the
// join stage stays exact in i32 (its terms reach 256 * 1024 * 512 < 2^31).

#include "common.hpp"

extern "C" void BENCH_FN(DTY *, DTY *, DTY *, DTY *, DTY *);

struct Mm3Diamond {
  static constexpr bench::Size kSizes[] = {
      {"_3mm_diam_d8", {8}},
      {"_3mm_diam_d16", {16}},
      {"_3mm_diam_d32", {32}},
      {"_3mm_diam_d64", {64}},
  };

  size_t d;
  std::vector<DTY> A, B, C, D, out;

  void setup(const size_t *dims) {
    d = dims[0];
    A = bench::random_vector(d * 1024, 2);
    B = bench::random_vector(1024 * 256, 2);
    C = bench::random_vector(256 * 512, 2);
    D = bench::random_vector(512 * 2048, 2);
    out = bench::output_vector(d * 2048);
    printf("%s  d=%zu", TOSTR(BENCH_FN), d);
  }

  void run() { BENCH_FN(A.data(), B.data(), C.data(), D.data(), out.data()); }
  const std::vector<DTY> &output() const { return out; }

  std::vector<double> reference() const {
    std::vector<double> l(d * 256), r(256 * 2048), j(d * 2048);
    bench::matmul_ref(A.data(), B.data(), d, 1024, 256, l.data());
    bench::matmul_ref(C.data(), D.data(), 256, 512, 2048, r.data());
    bench::matmul_ref(l.data(), r.data(), d, 256, 2048, j.data());
    return j;
  }
};

int main(int argc, char **argv) { return bench::run<Mm3Diamond>(argc, argv); }
