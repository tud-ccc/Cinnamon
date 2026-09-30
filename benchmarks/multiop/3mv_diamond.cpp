// j = (W1 * x) * (W2 * x), elementwise   (the diamond in vector form --
// llama's SwiGLU gate); x: K, W1, W2: MxK, j: M. Size class = bytes of each
// i32 weight (see 2mv_seq.cpp). The join is an elementwise product because
// a gemv cannot take a vector as its matrix, so two vector branches cannot
// close with another contraction.

#include "../common.hpp"

extern "C" void BENCH_FN(DTY *, DTY *, DTY *, DTY *);

struct Mv3Diamond {
  static constexpr bench::Size kSizes[] = {
      {"_3mv_diam_1MB", {512, 512}},
      {"_3mv_diam_64MB", {4096, 4096}},
      {"_3mv_diam_256MB", {8192, 8192}},
      {"_3mv_diam_512MB", {8192, 16384}},
  };

  size_t m, k;
  std::vector<DTY> x, W1, W2, out;

  void setup(const size_t *dims) {
    m = dims[0];
    k = dims[1];
    x = bench::random_vector(k);
    W1 = bench::random_vector(m * k);
    W2 = bench::random_vector(m * k);
    out = bench::output_vector(m);
    printf("%s  M=%zu K=%zu", TOSTR(BENCH_FN), m, k);
  }

  void run() { BENCH_FN(x.data(), W1.data(), W2.data(), out.data()); }
  const std::vector<DTY> &output() const { return out; }

  std::vector<double> reference() const {
    std::vector<uint32_t> l(m), r(m), j(m);
    bench::matmul_ref_wrap(W1.data(), x.data(), m, k, 1, l.data());
    bench::matmul_ref_wrap(W2.data(), x.data(), m, k, 1, r.data());
    for (size_t i = 0; i < m; i++)
      j[i] = l[i] * r[i]; // u32 wraparound, as on the device
    return bench::wrapped_golden(j);
  }
};

int main(int argc, char **argv) { return bench::run<Mv3Diamond>(argc, argv); }
