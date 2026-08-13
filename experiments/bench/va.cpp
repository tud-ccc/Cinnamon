// out = x + y   (elementwise add).

#include "common.hpp"

extern "C" void BENCH_FN(DTY *, DTY *, DTY *);

struct Va {
  static constexpr bench::Size kSizes[] = {
      {"va_4MB", {1048576}},
      {"va_64MB", {16777216}},
      {"va_256MB", {67108864}},
  };

  size_t n;
  std::vector<DTY> x, y, out;

  void setup(const size_t *d) {
    n = d[0];
    x = bench::random_vector(n);
    y = bench::random_vector(n);
    out.assign(n, 0);
    printf("%s  n=%zu", TOSTR(BENCH_FN), n);
  }

  void run() { BENCH_FN(x.data(), y.data(), out.data()); }
  const std::vector<DTY> &output() const { return out; }
  std::vector<double> reference() const {
    return bench::axpby_ref(x, y, 1.0, 1.0, n);
  }
};

int main(int argc, char **argv) { return bench::run<Va>(argc, argv); }
