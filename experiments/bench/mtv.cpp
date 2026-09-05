// out = A * x   (cinm.op.gemv), A is M×N row-major.

#include "common.hpp"

extern "C" void BENCH_FN(DTY *, DTY *, DTY *);

struct Mtv {
  static constexpr bench::Size kSizes[] = {
      {"mtv_4MB", {1024, 1024}},
      {"mtv_64MB", {4096, 4096}},
      {"mtv_256MB", {8192, 8192}},
      {"mtv_512MB", {8192, 16384}},
  };

  size_t m, n;
  std::vector<DTY> A, x, out;

  void setup(const size_t *d) {
    m = d[0];
    n = d[1];
    A = bench::random_vector(m * n);
    x = bench::random_vector(n);
    out = bench::output_vector(m);
    printf("%s  M=%zu N=%zu", TOSTR(BENCH_FN), m, n);
  }

  void run() { BENCH_FN(A.data(), x.data(), out.data()); }
  const std::vector<DTY> &output() const { return out; }
  std::vector<double> reference() const { return bench::gemv_ref(A, x, m, n); }
};

int main(int argc, char **argv) { return bench::run<Mtv>(argc, argv); }
