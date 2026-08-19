// out = (A * x) * c   (cinm.op.gemv followed by an elementwise scalar mul),
// A is M×N row-major.

#include "common.hpp"

extern "C" void BENCH_FN(DTY *, DTY *, DTY, DTY *);

struct Gemv {
  static constexpr bench::Size kSizes[] = {
      {"gemv_4MB", {1024, 1024}},
      {"gemv_64MB", {4096, 4096}},
      {"gemv_256MB", {8192, 8192}},
      {"gemv_512MB", {8192, 16384}},
  };

  size_t m, n;
  DTY c;
  std::vector<DTY> A, x, out;

  void setup(const size_t *d) {
    m = d[0];
    n = d[1];
    A = bench::random_vector(m * n);
    x = bench::random_vector(n);
    c = bench::next_operand();
    out = bench::output_vector(m);
    printf("%s  M=%zu N=%zu c=%lld", TOSTR(BENCH_FN), m, n, (long long)c);
  }

  void run() { BENCH_FN(A.data(), x.data(), c, out.data()); }
  const std::vector<DTY> &output() const { return out; }

  std::vector<double> reference() const {
    std::vector<double> r = bench::gemv_ref(A, x, m, n);
    for (double &v : r)
      v *= (double)c;
    return r;
  }
};

int main(int argc, char **argv) { return bench::run<Gemv>(argc, argv); }
