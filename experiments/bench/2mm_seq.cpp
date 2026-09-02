// r2 = (A * B) * C   (two chained cinm.op.gemm); A: d×1024, B: 1024×256,
// C: 256×2048. Inputs are 0/1 so both i32 stages stay exact (see the
// overflow note on bench::random_vector's bounded overload).

#include "common.hpp"

extern "C" void BENCH_FN(DTY *, DTY *, DTY *, DTY *);

struct Mm2Seq {
  static constexpr bench::Size kSizes[] = {
      {"_2mm_seq_d8", {8}},
      {"_2mm_seq_d16", {16}},
      {"_2mm_seq_d32", {32}},
      {"_2mm_seq_d64", {64}},
  };

  size_t d;
  std::vector<DTY> A, B, C, out;

  void setup(const size_t *dims) {
    d = dims[0];
    A = bench::random_vector(d * 1024, 2);
    B = bench::random_vector(1024 * 256, 2);
    C = bench::random_vector(256 * 2048, 2);
    out = bench::output_vector(d * 2048);
    printf("%s  d=%zu", TOSTR(BENCH_FN), d);
  }

  void run() { BENCH_FN(A.data(), B.data(), C.data(), out.data()); }
  const std::vector<DTY> &output() const { return out; }

  std::vector<double> reference() const {
    std::vector<double> t(d * 256), r(d * 2048);
    bench::matmul_ref(A.data(), B.data(), d, 1024, 256, t.data());
    bench::matmul_ref(t.data(), C.data(), d, 256, 2048, r.data());
    return r;
  }
};

int main(int argc, char **argv) { return bench::run<Mm2Seq>(argc, argv); }
