// result = sum(x)   (cinm.op.reduce add), returned by value.

#include "common.hpp"

extern "C" DTY BENCH_FN(DTY *);

struct Red {
  static constexpr bench::Size kSizes[] = {
      {"red_4MB", {524288}},
      {"red_64MB", {8388608}},
      {"red_256MB", {34554432}},
      {"red_512MB", {67108864}},
  };

  size_t n;
  std::vector<DTY> in;
  std::vector<DTY> result{0};

  void setup(const size_t *d) {
    n = d[0];
    in = bench::random_vector(n);
    printf("%s  n=%zu", TOSTR(BENCH_FN), n);
  }

  void run() { result[0] = BENCH_FN(in.data()); }
  const std::vector<DTY> &output() const { return result; }
  std::vector<double> reference() const { return {bench::sum_ref(in, n)}; }
};

int main(int argc, char **argv) { return bench::run<Red>(argc, argv); }
