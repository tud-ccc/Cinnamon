// (z, t) = (W2 * (W1 * x), W3 * x)   (a two-gemv chain plus one gemv
// parallel to it, sharing x); x: K, W1, W3: MxK, W2: KxM. Size class =
// bytes of each i32 weight (see 2mv_seq.cpp). Two out-params in result
// order, one backing vector (see 2mm_par.cpp).

#include "../common.hpp"

extern "C" void BENCH_FN(DTY *, DTY *, DTY *, DTY *, DTY *, DTY *);

struct Mv3ParSeq {
  static constexpr bench::Size kSizes[] = {
      {"_3mv_1MB", {512, 512}},
      {"_3mv_64MB", {4096, 4096}},
      {"_3mv_256MB", {8192, 8192}},
      {"_3mv_512MB", {8192, 16384}},
  };

  size_t m, k;
  std::vector<DTY> x, W1, W2, W3, out; // out = [z: k | t: m]

  void setup(const size_t *dims) {
    m = dims[0];
    k = dims[1];
    x = bench::random_vector(k);
    W1 = bench::random_vector(m * k);
    W2 = bench::random_vector(k * m);
    W3 = bench::random_vector(m * k);
    out = bench::output_vector(k + m);
    printf("%s  M=%zu K=%zu", TOSTR(BENCH_FN), m, k);
  }

  void run() {
    BENCH_FN(x.data(), W1.data(), W2.data(), W3.data(), out.data(),
             out.data() + k);
  }
  const std::vector<DTY> &output() const { return out; }

  std::vector<double> reference() const {
    std::vector<uint32_t> y(m), r(k + m);
    bench::matmul_ref_wrap(W1.data(), x.data(), m, k, 1, y.data());
    bench::matmul_ref_wrap(W2.data(), y.data(), k, m, 1, r.data());
    bench::matmul_ref_wrap(W3.data(), x.data(), m, k, 1, r.data() + k);
    return bench::wrapped_golden(r);
  }
};

int main(int argc, char **argv) { return bench::run<Mv3ParSeq>(argc, argv); }
