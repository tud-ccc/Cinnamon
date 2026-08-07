//===- PermutationTest.cpp - Permutation-valued search parameters --------===//
//
// A permutation crosses every interface here as an integer, so the properties
// worth testing are the ones that make that integer mean something: that the
// encoding round-trips, and that the two places which have to *interpret* a
// rank -- the surrogate's features and the neighbourhood -- do so in terms of
// the permutation rather than the number.
//
//===----------------------------------------------------------------------===//

#include "cinm-mlir/Utils/Permutation.h"
#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/SpaceBuilder.h"

#include <gtest/gtest.h>

#include <set>
#include <vector>

using namespace mlir::cinm;

namespace {

/// The permutation a one-based rank names, as SearchParam stores it.
std::vector<unsigned> decode(ParmValue rank, unsigned n) {
  llvm::SmallVector<unsigned> order = unrankPermutation(rank - 1, n);
  return {order.begin(), order.end()};
}

/// Number of positions in which two permutations of the same size differ.
unsigned positionsDiffering(llvm::ArrayRef<unsigned> a,
                            llvm::ArrayRef<unsigned> b) {
  unsigned n = 0;
  for (auto [x, y] : llvm::zip_equal(a, b))
    n += x != y;
  return n;
}

} // namespace

//===----------------------------------------------------------------------===//
// The encoding
//===----------------------------------------------------------------------===//

TEST(PermutationTest, RankRoundTrips) {
  for (unsigned n = 0; n <= 5; ++n) {
    const int64_t count = *factorial(n);
    std::set<std::vector<unsigned>> seen;
    for (int64_t rank = 0; rank < count; ++rank) {
      llvm::SmallVector<unsigned> order = unrankPermutation(rank, n);
      EXPECT_EQ(order.size(), n);
      EXPECT_EQ(rankPermutation(order), rank) << "at rank " << rank;
      seen.insert({order.begin(), order.end()});
    }
    // Every permutation exactly once: the ranks are a bijection, which is what
    // lets a rank stand in for a permutation at all.
    EXPECT_EQ(seen.size(), static_cast<size_t>(count));
  }
}

TEST(PermutationTest, RankZeroIsIdentity) {
  // The default rule sits at rank 0, which is what puts it at the origin of a
  // search space rather than somewhere in the middle of it.
  EXPECT_EQ(unrankPermutation(0, 4), (llvm::SmallVector<unsigned>{0, 1, 2, 3}));
}

//===----------------------------------------------------------------------===//
// What the surrogate sees
//===----------------------------------------------------------------------===//

TEST(PermutationTest, FeaturesArePositionsNotRank) {
  SearchParam param = makePermutation("order", 3);
  EXPECT_EQ(param.numFeatures(), 3u);

  // Feature i is where dimension i ended up, scaled to [0, 1]. Rank 1 is the
  // identity, so dimension i sits at axis i.
  llvm::SmallVector<double> features;
  param.appendFeatures({1}, features);
  EXPECT_EQ(features, (llvm::SmallVector<double>{0.0, 0.5, 1.0}));

  // Rank 2 is [0, 2, 1]: dimension 1 moved to the last axis and dimension 2 to
  // the middle one.
  features.clear();
  param.appendFeatures({2}, features);
  EXPECT_EQ(features, (llvm::SmallVector<double>{0.0, 1.0, 0.5}));
}

TEST(PermutationTest, FeatureDistanceIsSpearman) {
  const unsigned n = 4;
  SearchParam param = makePermutation("order", n);
  const int64_t count = *factorial(n);

  // Squared Euclidean distance between two position vectors is Spearman's rank
  // distance, up to the scaling. That equivalence is the whole reason the
  // features are positions, so it is worth pinning rather than assuming.
  const double scale = (n - 1) * (n - 1);
  for (int64_t p = 0; p < count; ++p) {
    for (int64_t q = 0; q < count; ++q) {
      llvm::SmallVector<double> fp, fq;
      param.appendFeatures({static_cast<ParmValue>(p + 1)}, fp);
      param.appendFeatures({static_cast<ParmValue>(q + 1)}, fq);

      double squared = 0;
      for (auto [x, y] : llvm::zip_equal(fp, fq))
        squared += (x - y) * (x - y);

      std::vector<unsigned> op = decode(p + 1, n), oq = decode(q + 1, n);
      std::vector<unsigned> posP(n), posQ(n);
      for (unsigned i = 0; i < n; ++i) {
        posP[op[i]] = i;
        posQ[oq[i]] = i;
      }
      double spearman = 0;
      for (unsigned i = 0; i < n; ++i) {
        double d = double(posP[i]) - double(posQ[i]);
        spearman += d * d;
      }
      EXPECT_DOUBLE_EQ(squared * scale, spearman);
    }
  }
}

//===----------------------------------------------------------------------===//
// What counts as a neighbour
//===----------------------------------------------------------------------===//

TEST(PermutationTest, NeighboursAreAdjacentTranspositions) {
  const unsigned n = 4;
  SearchParam param = makePermutation("order", n);

  for (int64_t rank = 0; rank < *factorial(n); ++rank) {
    llvm::SmallVector<llvm::SmallVector<ParmValue, 4>, 4> steps;
    param.appendNeighbours({static_cast<ParmValue>(rank + 1)}, steps);

    // One per adjacent pair, and each really is one swap away -- two positions
    // differing, next to each other.
    EXPECT_EQ(steps.size(), n - 1) << "at rank " << rank;
    std::vector<unsigned> from = decode(rank + 1, n);
    std::set<ParmValue> distinct;
    for (const auto &step : steps) {
      ASSERT_EQ(step.size(), 1u) << "a rank is one dimension";
      distinct.insert(step[0]);
      std::vector<unsigned> to = decode(step[0], n);
      EXPECT_EQ(positionsDiffering(from, to), 2u) << "at rank " << rank;
    }
    EXPECT_EQ(distinct.size(), steps.size());
  }
}

TEST(PermutationTest, NeighboursOfAQuantityAreAdjacentValues) {
  // The other kind, unchanged: a quantity steps to the next value its domain
  // holds, and the ends of the domain have one neighbour rather than two.
  SearchParam param = makeValues("tile", {1, 2, 4, 8});

  llvm::SmallVector<llvm::SmallVector<ParmValue, 4>, 4> steps;
  param.appendNeighbours({4}, steps);
  EXPECT_EQ(steps.size(), 2u);
  EXPECT_EQ(steps[0], (llvm::SmallVector<ParmValue, 4>{2}));
  EXPECT_EQ(steps[1], (llvm::SmallVector<ParmValue, 4>{8}));

  steps.clear();
  param.appendNeighbours({1}, steps);
  EXPECT_EQ(steps.size(), 1u);
  EXPECT_EQ(steps[0], (llvm::SmallVector<ParmValue, 4>{2}));
}

TEST(PermutationTest, NeighboursInASpaceAreReachableAndDistinct) {
  // End to end: a permutation parameter alongside a quantity, and every
  // neighbour the space reports must be a real configuration one step away.
  SpaceBuilder b;
  b.permutation("order", 3);
  b.divisorsOf("tile", 8);

  ConfigSpace space;
  b.buildInto(space);
  ASSERT_EQ(space.size(), 2u);

  Configuration conf;
  for (size_t i = 0; i < space.totalSize(); ++i) {
    space.at(i, conf);
    llvm::SmallVector<size_t> neighbours;
    space.neighborIndices(i, neighbours);

    std::set<size_t> distinct(neighbours.begin(), neighbours.end());
    EXPECT_EQ(distinct.size(), neighbours.size()) << "at index " << i;
    EXPECT_FALSE(distinct.count(i)) << "index " << i << " is its own neighbour";

    Configuration other;
    for (size_t j : neighbours) {
      ASSERT_LT(j, space.totalSize());
      space.at(j, other);
      // Exactly one parameter moved, and if it was the permutation it moved by
      // one transposition.
      unsigned changed = 0;
      for (size_t d = 0; d < space.size(); ++d)
        changed += conf[d] != other[d];
      EXPECT_EQ(changed, 1u) << "at index " << i;
      if (conf[0] != other[0])
        EXPECT_EQ(positionsDiffering(decode(conf[0], 3), decode(other[0], 3)),
                  2u);
    }
  }
}
