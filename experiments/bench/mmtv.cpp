// out[b][m] = sum_k A[b][m][k] * x[b][k]   (cinm.op.batch_gemv): one
// independent gemv per batch, each with its own vector.

#include "common.hpp"

extern "C" void BENCH_FN(DTY *, DTY *, DTY *);

struct Mmtv {
  static constexpr bench::Size kSizes[] = {
      {"mmtv_4MB", {32, 64, 512}},
      {"mmtv_64MB", {128, 256, 512}},
      {"mmtv_256MB", {256, 512, 512}},
      {"mmtv_512MB", {512, 512, 512}},
  };

  size_t b, m, k;
  std::vector<DTY> A, x, out;

  void setup(const size_t *d) {
    b = d[0];
    m = d[1];
    k = d[2];
    A = bench::random_vector(b * m * k);
    x = bench::random_vector(b * k);
    out = bench::output_vector(b * m);
    printf("%s  B=%zu M=%zu K=%zu", TOSTR(BENCH_FN), b, m, k);
  }

  void run() { BENCH_FN(A.data(), x.data(), out.data()); }
  const std::vector<DTY> &output() const { return out; }

  std::vector<double> reference() const {
    std::vector<double> r(b * m);
    for (size_t i = 0; i < b; i++) {
      std::vector<DTY> xi(x.begin() + i * k, x.begin() + (i + 1) * k);
      bench::gemv_ref(A.data() + i * m * k, xi, m, k, r.data() + i * m);
    }
    return r;
  }
};

int main(int argc, char **argv) { return bench::run<Mmtv>(argc, argv); }
