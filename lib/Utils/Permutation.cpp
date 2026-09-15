#include "cinm-mlir/Utils/Permutation.h"

#include <cassert>
#include <numeric>

namespace mlir::cinm {

std::optional<int64_t> factorial(unsigned n) {
  if (n > 20)
    return std::nullopt;
  int64_t result = 1;
  for (unsigned i = 2; i <= n; ++i)
    result *= i;
  return result;
}

llvm::SmallVector<unsigned> unrankPermutation(int64_t rank, unsigned n) {
  assert(rank >= 0 && (n == 0 || rank < *factorial(n)) && "rank out of range");

  llvm::SmallVector<unsigned> available(n);
  std::iota(available.begin(), available.end(), 0u);

  llvm::SmallVector<unsigned> permutation;
  permutation.reserve(n);
  for (unsigned remaining = n; remaining > 0; --remaining) {
    // The digit's weight is the number of permutations of the tail it leaves,
    // i.e. (remaining - 1)!.
    int64_t weight = 1;
    for (unsigned i = 2; i < remaining; ++i)
      weight *= i;
    auto digit = static_cast<size_t>(rank / weight);
    rank %= weight;
    permutation.push_back(available[digit]);
    available.erase(available.begin() + digit);
  }
  return permutation;
}

int64_t rankPermutation(llvm::ArrayRef<unsigned> permutation) {
  const unsigned n = permutation.size();
  llvm::SmallVector<unsigned> available(n);
  std::iota(available.begin(), available.end(), 0u);

  int64_t rank = 0;
  for (unsigned position = 0; position < n; ++position) {
    auto it = llvm::find(available, permutation[position]);
    assert(it != available.end() && "not a permutation of [0, n)");
    int64_t weight = 1;
    for (unsigned i = 2; i < n - position; ++i)
      weight *= i;
    rank += weight * static_cast<int64_t>(it - available.begin());
    available.erase(it);
  }
  return rank;
}

} // namespace mlir::cinm
