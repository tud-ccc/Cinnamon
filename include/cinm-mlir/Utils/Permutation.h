#pragma once

#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <optional>

namespace mlir::cinm {

/// Permutations of `[0, n)` as integers, in lexicographic order: the factorial
/// number system.
///
/// This encoding is how a permutation crosses an interface that carries only
/// numbers -- a search parameter, an attribute on an op -- so the two ends have
/// to agree on it exactly. It lives here rather than in either end because the
/// search framework and the lowering that reads what the search stamped are
/// otherwise independent of each other.
///
/// Rank 0 is the identity, which is what puts a default rule at the origin of a
/// search space. Note that the *search parameter* holding a rank is one-based,
/// like every other search parameter; the conversion is at that boundary, not
/// here.

/// `n!`, or nullopt if it does not fit in an int64_t (n > 20).
std::optional<int64_t> factorial(unsigned n);

/// The `rank`-th permutation of `[0, n)`. `rank` must be in `[0, n!)`.
llvm::SmallVector<unsigned> unrankPermutation(int64_t rank, unsigned n);

/// Inverse of unrankPermutation: the lexicographic rank of `permutation`,
/// which must be a permutation of `[0, permutation.size())`.
int64_t rankPermutation(llvm::ArrayRef<unsigned> permutation);

} // namespace mlir::cinm
