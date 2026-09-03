// j = (A * B) * (C * D)   (diamond: two independent gemms join in a third);
// A: 8×N, B: N×N, C: N×N, D: N×8, j: 8×8. Size class = bytes of each N×N
// i32 weight (see 2mm_seq.cpp); the statics are the inner factors B and C,
// which keeps both intermediates skinny (8×N and N×8). References wrap in
// u32 like the device's i32 (see matmul_ref_wrap).

#include "common.hpp"

extern "C" void BENCH_FN(DTY *, DTY *, DTY *, DTY *, DTY *);

struct Mm3Diamond {
  static constexpr bench::Size kSizes[] = {
      {"_3mm_diam_1MB", {512}},
      {"_3mm_diam_16MB", {2048}},
      {"_3mm_diam_64MB", {4096}},
      {"_3mm_diam_256MB", {8192}},
  };

  static constexpr size_t d = 8;
  size_t n;
  std::vector<DTY> A, B, C, D, out;

  void setup(const size_t *dims) {
    n = dims[0];
    A = bench::random_vector(d * n);
    B = bench::random_vector(n * n);
    C = bench::random_vector(n * n);
    D = bench::random_vector(n * d);
    out = bench::output_vector(d * d);
    printf("%s  N=%zu", TOSTR(BENCH_FN), n);
  }

  void run() { BENCH_FN(A.data(), B.data(), C.data(), D.data(), out.data()); }
  const std::vector<DTY> &output() const { return out; }

  std::vector<double> reference() const {
    std::vector<uint32_t> l(d * n), r(n * d), j(d * d);
    bench::matmul_ref_wrap(A.data(), B.data(), d, n, n, l.data());
    bench::matmul_ref_wrap(C.data(), D.data(), n, n, d, r.data());
    bench::matmul_ref_wrap(l.data(), r.data(), d, n, d, j.data());
    return bench::wrapped_golden(j);
  }
};

int main(int argc, char **argv) { return bench::run<Mm3Diamond>(argc, argv); }
