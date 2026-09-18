// (y1, y2) = (W1 * x, W2 * x)   (two independent cinm.op.gemv sharing x --
// llama's QKV projections); x: K, W1, W2: MxK. Size class = bytes of each
// i32 weight (see 2mv_seq.cpp). Two results, so two out-pointers in result
// order into one backing vector (see 2mm_par.cpp).

#include "../common.hpp"

extern "C" void BENCH_FN(DTY *, DTY *, DTY *, DTY *, DTY *);

struct Mv2Par {
  static constexpr bench::Size kSizes[] = {
      {"_2mv_par_1MB", {512, 512}},
      {"_2mv_par_64MB", {4096, 4096}},
      {"_2mv_par_256MB", {8192, 8192}},
      {"_2mv_par_512MB", {8192, 16384}},
  };

  size_t m, k;
  std::vector<DTY> x, W1, W2, out; // out = [y1: m | y2: m]

  void setup(const size_t *dims) {
    m = dims[0];
    k = dims[1];
    x = bench::random_vector(k);
    W1 = bench::random_vector(m * k);
    W2 = bench::random_vector(m * k);
    out = bench::output_vector(2 * m);
    printf("%s  M=%zu K=%zu", TOSTR(BENCH_FN), m, k);
  }

  void run() {
    BENCH_FN(x.data(), W1.data(), W2.data(), out.data(), out.data() + m);
  }
  const std::vector<DTY> &output() const { return out; }

  std::vector<double> reference() const {
    std::vector<uint32_t> r(2 * m);
    bench::matmul_ref_wrap(W1.data(), x.data(), m, k, 1, r.data());
    bench::matmul_ref_wrap(W2.data(), x.data(), m, k, 1, r.data() + m);
    return bench::wrapped_golden(r);
  }
};

int main(int argc, char **argv) { return bench::run<Mv2Par>(argc, argv); }
