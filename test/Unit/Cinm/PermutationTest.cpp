//===- PermutationTest.cpp - Permutation-valued search parameters --------===//
//
// An ordering is stored positionally -- one dimension per item, holding the
// place it takes -- so the properties worth testing are the ones that make
// those n numbers an ordering rather than n independent choices: that only
// distinct assignments survive, that a caller reading one back gets places and
// not the encoding, and that the two derived views (the surrogate's features
// and the neighbourhood) are stated in terms of the ordering.
//
// The factorial number system is tested here too, but for a different reason:
// it is no longer how a parameter is stored, only how an order crosses into
// `cnm.workgroup_dim_order_index`. Its round trip is what that boundary rests
// on.
//
//===----------------------------------------------------------------------===//

#include "cinm-mlir/Utils/Permutation.h"
#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/SpaceBuilder.h"

#include <gtest/gtest.h>

#include <set>
#include <vector>

using namespace mlir::cinm;

namespace {

using Places = llvm::SmallVector<ParmValue, 4>;

/// The places of an ordering, as the encoding holds them: one-based, item i at
/// index i.
Places places(std::initializer_list<ParmValue> p) { return Places(p); }

/// Number of items whose place differs between two orderings.
unsigned itemsMoved(llvm::ArrayRef<ParmValue> a, llvm::ArrayRef<ParmValue> b) {
  unsigned n = 0;
  for (auto [x, y] : llvm::zip_equal(a, b))
    n += x != y;
  return n;
}

} // namespace

//===----------------------------------------------------------------------===//
// The encoding
//===----------------------------------------------------------------------===//

TEST(PermutationTest, ParameterSpansOneDimensionPerItem) {
  SearchParam param = makePermutation("order", 3);
  EXPECT_EQ(param.arity(), 3u);
  EXPECT_EQ(param.kind(), ParamKind::Permutation);
  // Every dimension offers every place; distinctness is the solver's job, not
  // the domain's.
  EXPECT_EQ(param.cardinality(), 3u);
  EXPECT_EQ(param.dlo(), 1.0);
  EXPECT_EQ(param.dhi(), 3.0);
  // ...so the domain is not how many values the parameter has. Three
  // dimensions of three places span 27 encodings, of which distinctness keeps
  // the 3! that are orderings. Neither 3 nor 27 is the answer; a space size
  // built from either is wrong, in opposite directions.
  EXPECT_EQ(param.numValues(), 6.0);
  // A quantity occupies one dimension, so for it the two do coincide.
  EXPECT_EQ(makeRange("tile", 1, 8).numValues(), 8.0);
  EXPECT_EQ(makeRange("tile", 1, 8).cardinality(), 8u);
  // A parameter of arity one is named for itself; these are not.
  EXPECT_EQ(param.dimName(0), "order[0]");
  EXPECT_EQ(makeRange("tile", 1, 8).dimName(0), "tile");
}

TEST(PermutationTest, DecodeIsZeroBasedPlaces) {
  SearchParam param = makePermutation("order", 3);
  // The parameter is one-based, like every other; Permutation is not. The
  // offset is the model's business and shows up nowhere else.
  Permutation perm = ParmKind<Permutation>::decode(param, places({1, 3, 2}));
  EXPECT_EQ(perm.size(), 3u);
  EXPECT_EQ(perm[0], 0u);
  EXPECT_EQ(perm[1], 2u);
  EXPECT_EQ(perm[2], 1u);
}

//===----------------------------------------------------------------------===//
// What the surrogate sees
//===----------------------------------------------------------------------===//

TEST(PermutationTest, FeaturesArePlaces) {
  SearchParam param = makePermutation("order", 3);
  EXPECT_EQ(param.numFeatures(), 3u);

  // Feature i is where item i ended up, scaled to [0, 1].
  llvm::SmallVector<double> features;
  param.appendFeatures(places({1, 2, 3}), features);
  EXPECT_EQ(features, (llvm::SmallVector<double>{0.0, 0.5, 1.0}));

  features.clear();
  param.appendFeatures(places({1, 3, 2}), features);
  EXPECT_EQ(features, (llvm::SmallVector<double>{0.0, 1.0, 0.5}));
}

TEST(PermutationTest, FeatureDistanceIsSpearman) {
  const unsigned n = 4;
  SearchParam param = makePermutation("order", n);

  // Squared Euclidean distance between two feature vectors is Spearman's rank
  // distance, up to the scaling. That equivalence is the whole reason the
  // features are places, so it is worth pinning rather than assuming.
  const double scale = (n - 1) * (n - 1);
  for (int64_t p = 0; p < *factorial(n); ++p) {
    for (int64_t q = 0; q < *factorial(n); ++q) {
      Places pp, pq;
      for (unsigned item : unrankPermutation(p, n))
        pp.push_back(static_cast<ParmValue>(item) + 1);
      for (unsigned item : unrankPermutation(q, n))
        pq.push_back(static_cast<ParmValue>(item) + 1);

      llvm::SmallVector<double> fp, fq;
      param.appendFeatures(pp, fp);
      param.appendFeatures(pq, fq);

      double squared = 0;
      for (auto [x, y] : llvm::zip_equal(fp, fq))
        squared += (x - y) * (x - y);

      double spearman = 0;
      for (auto [x, y] : llvm::zip_equal(pp, pq))
        spearman += double(x - y) * double(x - y);

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
    Places from;
    for (unsigned item : unrankPermutation(rank, n))
      from.push_back(static_cast<ParmValue>(item) + 1);

    llvm::SmallVector<Places, 4> steps;
    param.appendNeighbours(from, steps);

    // One per adjacent pair of places, and each really is one swap away: two
    // items moved, and to each other's place.
    EXPECT_EQ(steps.size(), n - 1) << "at rank " << rank;
    std::set<Places> distinct;
    for (const Places &to : steps) {
      ASSERT_EQ(to.size(), n) << "a step assigns the whole parameter";
      distinct.insert(to);
      EXPECT_EQ(itemsMoved(from, to), 2u) << "at rank " << rank;
      // Still an ordering: the places are a permutation of 1..n.
      std::set<ParmValue> seen(to.begin(), to.end());
      EXPECT_EQ(seen.size(), n);
    }
    EXPECT_EQ(distinct.size(), steps.size());
  }
}

TEST(PermutationTest, NeighboursOfAQuantityAreAdjacentValues) {
  // The other kind, unchanged: a quantity steps to the next value its domain
  // holds, and the ends of the domain have one neighbour rather than two.
  SearchParam param = makeValues("tile", {1, 2, 4, 8});

  llvm::SmallVector<Places, 4> steps;
  param.appendNeighbours({4}, steps);
  EXPECT_EQ(steps.size(), 2u);
  EXPECT_EQ(steps[0], places({2}));
  EXPECT_EQ(steps[1], places({8}));

  steps.clear();
  param.appendNeighbours({1}, steps);
  EXPECT_EQ(steps.size(), 1u);
  EXPECT_EQ(steps[0], places({2}));
}

//===----------------------------------------------------------------------===//
// The parameter in a space
//===----------------------------------------------------------------------===//

TEST(PermutationTest, DistinctnessIsPostedByTheSolver) {
  // Nothing below declares it, and no caller ever will: it follows from the
  // parameter's kind (see ConstraintGecode.cpp). Without it this space would
  // hold 3^3 = 27 assignments of the ordering instead of 3! = 6.
  SpaceBuilder b;
  b.permutation("order", 3);

  ConfigSpace space;
  b.buildInto(space);
  EXPECT_EQ(space.numParams(), 1u);
  EXPECT_EQ(space.numDims(), 3u);
  EXPECT_EQ(space.totalSize(), 6u);

  std::set<Places> seen;
  Configuration conf;
  for (size_t i = 0; i < space.totalSize(); ++i) {
    space.at(i, conf);
    seen.insert(Places(conf.begin(), conf.end()));
  }
  EXPECT_EQ(seen.size(), 6u);
}

TEST(PermutationTest, SpaceSizeIsAProductOfValuesNotOfDomains) {
  // A space is never larger than the values its parameters range over, which
  // is the invariant a reported "Cartesian size" has to satisfy to be one.
  // Both ways of building that size out of `cardinality()` break it here, in
  // opposite directions, which is why numValues() exists.
  SpaceBuilder b;
  b.permutation("order", 3);
  b.intRange("tile", 1, 4);

  ConfigSpace space;
  b.buildInto(space);
  EXPECT_EQ(space.totalSize(), 6u * 4u);

  double values = 1, perParamDomain = 1, perDimDomain = 1;
  for (const SearchParam &p : space.params) {
    values *= p.numValues();
    perParamDomain *= static_cast<double>(p.cardinality());
    for (size_t k = 0, e = p.arity(); k < e; ++k)
      perDimDomain *= static_cast<double>(p.cardinality());
  }

  // 3! * 4.
  EXPECT_EQ(values, 24.0);
  EXPECT_LE(static_cast<double>(space.totalSize()), values);

  // Per parameter the ordering counts once (3, not 3!), so the "size" comes
  // out *below* the number of configurations and the density exceeds 1.
  EXPECT_EQ(perParamDomain, 12.0);
  EXPECT_LT(perParamDomain, static_cast<double>(space.totalSize()));

  // Per dimension it counts 3^3, so the space looks 27/6 = 4.5x sparser than
  // the constraints made it -- credit for distinctness, which no caller wrote.
  EXPECT_EQ(perDimDomain, 108.0);
  EXPECT_DOUBLE_EQ(perDimDomain / values, 4.5);
}

TEST(PermutationTest, InactiveItemsTakeTheHighPlacesInIndexOrder) {
  // The overload that says which items this configuration actually orders.
  // `n` is a quantity, and item i is active iff i < n, so a configuration with
  // n = k has exactly k! orderings -- the inactive tail is pinned rather than
  // free, or the same choice would appear once per rearrangement of items that
  // take no place at all.
  SpaceBuilder b;
  IntVar n = b.intRange("n", 1, 3);
  llvm::SmallVector<BoolExpr> active;
  for (ParmValue i = 0; i < 3; ++i)
    active.push_back(n > i);
  PermVar order = b.permutation("order", active);

  ConfigSpace space;
  b.buildInto(space);
  // 1! + 2! + 3! = 9.
  EXPECT_EQ(space.totalSize(), 9u);

  Configuration conf;
  for (size_t i = 0; i < space.totalSize(); ++i) {
    space.at(i, conf);
    ConfWrapper c(space, conf);
    const ParmValue count = c["n"];
    Permutation perm = order.get(c);
    for (unsigned item = 0; item < 3; ++item) {
      if (item < static_cast<unsigned>(count))
        EXPECT_LT(perm[item], static_cast<unsigned>(count))
            << "active item " << item << " took a place past the active ones";
      else
        EXPECT_EQ(perm[item], item)
            << "an inactive item moved out of index order";
    }
  }
}

TEST(PermutationTest, NeighboursInASpaceAreReachableAndDistinct) {
  // End to end: an ordering alongside a quantity, and every neighbour the
  // space reports must be a real configuration one step away.
  SpaceBuilder b;
  b.permutation("order", 3);
  b.divisorsOf("tile", 8);

  ConfigSpace space;
  b.buildInto(space);
  ASSERT_EQ(space.numParams(), 2u);
  ASSERT_EQ(space.numDims(), 4u);

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
      // Exactly one *parameter* moved, which for the ordering means two of its
      // dimensions at once -- a step in one alone would name no ordering.
      const unsigned orderMoved =
          itemsMoved(llvm::ArrayRef(conf).take_front(3),
                     llvm::ArrayRef(other).take_front(3));
      const bool tileMoved = conf[3] != other[3];
      if (tileMoved)
        EXPECT_EQ(orderMoved, 0u) << "at index " << i;
      else
        EXPECT_EQ(orderMoved, 2u) << "at index " << i;
    }
  }
}

//===----------------------------------------------------------------------===//
// The factorial number system
//===----------------------------------------------------------------------===//
//
// No longer an encoding of a search parameter -- it is how an order reaches
// `cnm.workgroup_dim_order_index`, which is a rank because it is stated over a
// list of dimensions that does not exist until the lowering builds it.

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
