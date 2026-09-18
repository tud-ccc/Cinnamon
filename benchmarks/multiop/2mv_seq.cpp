// z = W2 * (W1 * x)   (two chained cinm.op.gemv); x: K, W1: MxK, W2: KxM.
// The size class names the bytes of each i32 weight, both holding M*K
// elements: 1MB (512x512), 64MB (4096x4096), 256MB (8192x8192), 512MB
// (8192x16384, prim_mtv's 512MB shape). A gemv is matmul_ref_wrap with one
// output column, so the reference wraps in u32 like the device's i32 and
// operands use the default range (see 2mm_seq.cpp).

#include "../common.hpp"

extern "C" void BENCH_FN(DTY *, DTY *, DTY *, DTY *);

struct Mv2Seq {
  static constexpr bench::Size kSizes[] = {
      {"_2mv_seq_1MB", {512, 512}},
      {"_2mv_seq_64MB", {4096, 4096}},
      {"_2mv_seq_256MB", {8192, 8192}},
      {"_2mv_seq_512MB", {8192, 16384}},
  };

  size_t m, k;
  std::vector<DTY> x, W1, W2, out;

  void setup(const size_t *dims) {
    m = dims[0];
    k = dims[1];
    x = bench::random_vector(k);
    W1 = bench::random_vector(m * k); // MxK, y = W1 x  (length m)
    W2 = bench::random_vector(k * m); // KxM, z = W2 y  (length k)
    out = bench::output_vector(k);
    printf("%s  M=%zu K=%zu", TOSTR(BENCH_FN), m, k);
  }

  void run() { BENCH_FN(x.data(), W1.data(), W2.data(), out.data()); }
  const std::vector<DTY> &output() const { return out; }

  std::vector<double> reference() const {
    std::vector<uint32_t> y(m), z(k);
    bench::matmul_ref_wrap(W1.data(), x.data(), m, k, 1, y.data());
    bench::matmul_ref_wrap(W2.data(), y.data(), k, m, 1, z.data());
    return bench::wrapped_golden(z);
  }
};

int main(int argc, char **argv) { return bench::run<Mv2Seq>(argc, argv); }
