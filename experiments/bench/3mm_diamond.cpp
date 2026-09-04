// j = (A * B) * (C * D)   (diamond: two independent gemms join in a third);
// A: 8xK, B: KxM, C: MxK, D: Kx8, j: 8x8. Size class = bytes of each i32
// weight, both holding K*M elements (see 2mm_seq.cpp); the statics are the
// inner factors B and C, which keeps both intermediates skinny (8xM and
// Mx8). References wrap in u32 like the device's i32 (see matmul_ref_wrap).

#include "common.hpp"

extern "C" void BENCH_FN(DTY *, DTY *, DTY *, DTY *, DTY *);

struct Mm3Diamond {
  static constexpr bench::Size kSizes[] = {
      {"_3mm_diam_1MB", {512, 512}},
      {"_3mm_diam_64MB", {4096, 4096}},
      {"_3mm_diam_256MB", {8192, 8192}},
      {"_3mm_diam_512MB", {8192, 16384}},
  };

  static constexpr size_t d = 8;
  size_t k, m;
  std::vector<DTY> A, B, C, D, out;

  void setup(const size_t *dims) {
    k = dims[0];
    m = dims[1];
    A = bench::random_vector(d * k);
    B = bench::random_vector(k * m);
    C = bench::random_vector(m * k);
    D = bench::random_vector(k * d);
    out = bench::output_vector(d * d);
    printf("%s  K=%zu M=%zu", TOSTR(BENCH_FN), k, m);
  }

  void run() { BENCH_FN(A.data(), B.data(), C.data(), D.data(), out.data()); }
  const std::vector<DTY> &output() const { return out; }

  std::vector<double> reference() const {
    std::vector<uint32_t> l(d * m), r(m * d), j(d * d);
    bench::matmul_ref_wrap(A.data(), B.data(), d, k, m, l.data());
    bench::matmul_ref_wrap(C.data(), D.data(), m, k, d, r.data());
    bench::matmul_ref_wrap(l.data(), r.data(), d, m, d, j.data());
    return bench::wrapped_golden(j);
  }
};

int main(int argc, char **argv) { return bench::run<Mm3Diamond>(argc, argv); }
