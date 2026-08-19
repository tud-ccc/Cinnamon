// out = c*x + d*y   (two elementwise scalar muls and an elementwise add).

#include "common.hpp"

extern "C" void BENCH_FN(DTY *, DTY *, DTY, DTY, DTY *);

struct Geva {
  static constexpr bench::Size kSizes[] = {
      {"geva_4MB", {1048576}},
      {"geva_64MB", {16777216}},
      {"geva_256MB", {67108864}},
  };

  size_t n;
  DTY c, d;
  std::vector<DTY> x, y, out;

  void setup(const size_t *dims) {
    n = dims[0];
    x = bench::random_vector(n);
    y = bench::random_vector(n);
    c = bench::next_operand();
    d = bench::next_operand();
    out = bench::output_vector(n);
    printf("%s  n=%zu c=%lld d=%lld", TOSTR(BENCH_FN), n, (long long)c,
           (long long)d);
  }

  void run() { BENCH_FN(x.data(), y.data(), c, d, out.data()); }
  const std::vector<DTY> &output() const { return out; }
  std::vector<double> reference() const {
    return bench::axpby_ref(x, y, (double)c, (double)d, n);
  }
};

int main(int argc, char **argv) { return bench::run<Geva>(argc, argv); }
