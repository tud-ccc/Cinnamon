//===- SplitDerivationTest.cpp - Deriving the space from the schedule ----===//
//
// Experiment 1 for schedule-derived design spaces: can the constraint system
// be derived by folding a per-op contribution along a schedule, when one of
// the ops rewrites the iteration space in a way that depends on the parameter
// values?
//
// The op in question is `--convert-linalg-to-cnm`. A reduction dimension whose
// block size is smaller than its extent is *split*: the op is rewritten into a
// partial reduction plus a host-side merge, and a new iteration dimension is
// prepended (LinalgToCnm.cpp, splitDistributedReductions). How many dimensions
// that adds is a property of the configuration, so the rank of the iteration
// space downstream of the distribution is not known when the space is
// declared. A symbolic domain cannot have a configuration-dependent rank, so
// either the derivation models the split or it is unsound.
//
// What an op's interface emits is its *preconditions*, reified over the knobs
// it consumes: the configurations for which the op would refuse to run. That
// is the whole of the constraint system, not a category within it. The
// distribution below posts four, and each is a condition the pass really
// checks -- that the tile counts fill the workgroup, that the operand tiles
// fit, that no reduction it must split has a parallel dimension after it, and
// that a float reduction is not split unless reassociation was opted into.
//
// The three propositions under test are not constraints. They are the
// soundness argument for the *transfer function*: why a precondition stated
// over the pre-split iteration space still means what it says once the split
// has changed that space underneath it.
//
//   H1 (fill)       prod over dims of (extent / block) is unchanged, because
//                   a split dimension has `ratio` tiles and the reduction it
//                   came from is left spanning exactly one block.
//   H2 (footprint)  every operand's tile footprint is unchanged, because a
//                   split dimension is given tile size 1 at every inner level
//                   (that is what the pass's per-dim-attribute carrying does)
//                   and so contributes a factor of 1 to every operand it is
//                   added to.
//   H3 (order)      the number of dimensions actually spread over the
//                   workgroup is unchanged, so the ordering parameter has the
//                   same number of active items and hence the same number of
//                   assignments per configuration.
//
// If they hold, the transfer function of a distribution is just
// `extents := blocks`, with the operand indexing and the iterator kinds
// carried through unchanged, and the whole rank problem disappears.
//
// Note which of the two preconditions about splitting comes from the op's
// *structure* and which from its *options*. The parallel-first one is read off
// the iteration space; the float one is read off `allow-float-reassociation`,
// the same field the rewrite reads. An interface on the op cannot disagree
// with the rewrite about the second, which is exactly what the hand-written
// plugin does today: it constructs the pass without the option and declares no
// corresponding constraint, so on a float workload it proposes configurations
// that cannot lower. See FloatReassociationGate.
//
// Method: differential test. One side is a concrete reference that simulates
// the split in integer arithmetic, configuration by configuration, and
// evaluates the constraints on the iteration space the pass would really
// produce. The other is the compositional model below, which declares
// everything symbolically over the *pre*-split space and never represents a
// split at all. The two accepted sets must be equal.
//
// Equality is the right test only because every precondition modelled here is
// *exact* -- the op fails iff the constraint is violated. A precondition the
// derivation can only approximate needs containment instead: the real capacity
// test is --upmem-check-occupancy, which measures the lowered program rather
// than a formula, and the footprint bound is a deliberate over-rejection of it
// (UpmemInferAccelerator.cpp:829). This harness cannot see that gap, because
// the reference here evaluates the same formula the model declares. Closing it
// needs the pipeline.
//
//===----------------------------------------------------------------------===//

#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/SpaceBuilder.h"

#include <gtest/gtest.h>

#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

using namespace mlir::cinm;

namespace {

// ===----------------------------------------------------------------------===//
// The op, as the derivation needs to see it
// ===----------------------------------------------------------------------===//

/// Everything a constraint contribution reads off a linalg op: the iteration
/// space, which dimensions each operand is indexed by, and which dimensions
/// are reductions. This is `linalgLoopExtents` + `linalgOperandDims` +
/// `getIteratorTypesArray` from the UPMEM plugin, with no MLIR in the way.
struct OpDesc {
  std::string name;
  std::vector<int64_t> extents;
  std::vector<bool> isReduction;
  /// Iteration dimensions each operand is indexed by. Inputs first, then the
  /// DPS inits; only the *set* matters, since every use of it is a product.
  std::vector<std::vector<unsigned>> operandDims;
  unsigned numInputs;
  /// Splitting a float reduction reassociates it, which the distribution
  /// refuses unless its `allow-float-reassociation` option says otherwise.
  bool elementIsFloat = false;
};

// ===----------------------------------------------------------------------===//
// Reference: the split, simulated concretely
// ===----------------------------------------------------------------------===//

/// The iteration space at some point in the schedule, with sizes as integers.
struct ConcreteSpace {
  std::vector<int64_t> extents;
  /// Tile size per dimension, one list per memory level, outermost first.
  /// `sizes[0]` is the workgroup distribution.
  std::vector<std::vector<int64_t>> sizes;
  std::vector<bool> isReduction;
  std::vector<std::vector<unsigned>> operandDims;
  unsigned numInputs = 0;
  /// False when the pass would have refused this configuration outright.
  bool legal = true;
};

/// A faithful transcription of splitDistributedReductions (LinalgToCnm.cpp:186)
/// including the bookkeeping the pass does on the per-dimension attribute
/// lists: `blocks` gets a 1 inserted for the new dimension, and so does every
/// list in `per-dim-attrs` -- which is how the leaf tile sizes follow the
/// split.
ConcreteSpace applySplits(const OpDesc &op,
                          const std::vector<std::vector<int64_t>> &sizes,
                          bool allowFloatReassociation) {
  ConcreteSpace s{op.extents,     sizes,        op.isReduction,
                  op.operandDims, op.numInputs, true};

  while (true) {
    // Find the first reduction dimension the block sizes ask to spread. The
    // pass refuses an op whose parallel dimensions do not all come first,
    // because the insert position below assumes they do.
    std::optional<unsigned> target;
    for (unsigned d = 0; d < s.extents.size(); ++d) {
      if (!s.isReduction[d]) {
        if (target) {
          s.legal = false;
          return s;
        }
        continue;
      }
      if (!target && s.extents[d] != s.sizes[0][d])
        target = d;
    }
    if (!target)
      return s;

    if (s.sizes[0][*target] == 0 ||
        s.extents[*target] % s.sizes[0][*target] != 0) {
      s.legal = false;
      return s;
    }
    // Reassociating a float reduction changes the result, so it is opt-in.
    if (op.elementIsFloat && !allowFloatReassociation) {
      s.legal = false;
      return s;
    }
    const int64_t ratio = s.extents[*target] / s.sizes[0][*target];

    // The rewritten iteration space is [split dim] ++ [old dims], with the
    // split dimension holding one tile per leaf and the original reduction
    // dimension now spanning exactly one block.
    s.extents.insert(s.extents.begin(), ratio);
    // Every list indexed by iteration dimension gets a 1 for the new one --
    // `cnm.tile_sizes` and every name in `per-dim-attrs` alike.
    for (std::vector<int64_t> &level : s.sizes)
      level.insert(level.begin(), 1);
    s.isReduction.insert(s.isReduction.begin(), false);

    const unsigned moved = *target + 1;
    s.extents[moved] = s.sizes[0][moved];

    // Operands indexed by the split reduction have that dimension expanded
    // into (ratio, extent/ratio), so they gain the new dimension; so does
    // every init, which now holds one partial result per leaf.
    for (unsigned i = 0; i < s.operandDims.size(); ++i) {
      std::vector<unsigned> &dims = s.operandDims[i];
      bool indexedBySplit = false;
      for (unsigned &d : dims) {
        if (d == *target)
          indexedBySplit = true;
        d += 1;
      }
      if (indexedBySplit || i >= s.numInputs)
        dims.insert(dims.begin(), 0);
    }
  }
}

/// One operand's tile footprint at a level, in elements.
int64_t operandFootprint(const std::vector<unsigned> &dims,
                         const std::vector<int64_t> &sizes) {
  int64_t product = 1;
  for (unsigned d : dims)
    product *= sizes[d];
  return product;
}

int64_t footprint(const ConcreteSpace &s, const std::vector<int64_t> &sizes,
                  int64_t tasklets) {
  int64_t total = 0;
  for (const auto &dims : s.operandDims)
    total += operandFootprint(dims, sizes);
  return tasklets * total;
}

/// How many dimensions this configuration actually spreads over the workgroup.
/// A dimension cut into one tile takes no workgroup axis.
unsigned activeAxes(const ConcreteSpace &s) {
  unsigned count = 0;
  for (unsigned d = 0; d < s.extents.size(); ++d)
    if (s.extents[d] / s.sizes[0][d] > 1)
      count += 1;
  return count;
}

struct Platform {
  /// Capacity in elements per leaf, one per memory level, outermost first.
  /// Its length is how many levels the schedule tiles for: one distribution
  /// plus one staging step per level below it.
  std::vector<int64_t> capacities;
  int64_t maxDpus;
  int64_t maxTasklets;
  /// An *option* of the distribution op, not a search parameter: the interface
  /// implementation and the rewrite read the same field, so they cannot drift.
  bool allowFloatReassociation = false;
  size_t numLevels() const { return capacities.size(); }
};

/// The reference verdict: run the split, then evaluate on the space the pass
/// really produces.
bool concreteAccepts(const OpDesc &op, const Platform &plat,
                     const std::vector<std::vector<int64_t>> &sizes,
                     int64_t dpus, int64_t tasklets,
                     unsigned *activeOut = nullptr) {
  // Preconditions the distribution checks before it splits anything: the tile
  // sizes must divide the extents, and each level's tile must divide the one
  // above it.
  for (unsigned d = 0; d < op.extents.size(); ++d) {
    if (sizes[0][d] < 1 || op.extents[d] % sizes[0][d] != 0)
      return false;
    for (size_t l = 1; l < sizes.size(); ++l)
      if (sizes[l][d] < 1 || sizes[l - 1][d] % sizes[l][d] != 0)
        return false;
  }

  ConcreteSpace s = applySplits(op, sizes, plat.allowFloatReassociation);
  if (!s.legal)
    return false;

  int64_t tiles = 1;
  for (unsigned d = 0; d < s.extents.size(); ++d) {
    if (s.extents[d] % s.sizes[0][d] != 0)
      return false;
    tiles *= s.extents[d] / s.sizes[0][d];
  }
  if (tiles != dpus * tasklets)
    return false;

  for (size_t l = 0; l < s.sizes.size(); ++l)
    if (footprint(s, s.sizes[l], tasklets) > plat.capacities[l])
      return false;

  if (activeOut)
    *activeOut = activeAxes(s);
  return true;
}

// ===----------------------------------------------------------------------===//
// The compositional model
// ===----------------------------------------------------------------------===//
//
// One contribution per schedule op. Each declares the knobs it consumes,
// posts the constraints they must satisfy, and returns the abstract state its
// output is in. Nothing here knows what the *next* op will be, and nothing
// represents a split.

/// The abstract payload state. An extent is a constant at the top of the
/// schedule and a knob once a distribution has quotiented it, which is the
/// only thing the domain has to be polymorphic in.
struct AbstractOp {
  struct Size {
    ParmValue constant = 0;
    std::optional<IntVar> var;
    IntExpr expr() const { return var ? IntExpr(*var) : IntExpr(constant); }
  };
  std::vector<Size> extents;
  std::vector<bool> isReduction;
  std::vector<std::vector<unsigned>> operandDims;
  unsigned numInputs = 0;
  bool elementIsFloat = false;
};

AbstractOp initialState(const OpDesc &op) {
  AbstractOp st;
  for (int64_t e : op.extents)
    st.extents.push_back({static_cast<ParmValue>(e), std::nullopt});
  st.isReduction = op.isReduction;
  st.operandDims = op.operandDims;
  st.numInputs = op.numInputs;
  st.elementIsFloat = op.elementIsFloat;
  return st;
}

/// Declare one tile size per iteration dimension, each dividing the extent it
/// cuts -- whether that extent is a constant or the tile size the level above
/// chose.
std::vector<IntVar> declareTiles(const AbstractOp &in,
                                 const std::string &prefix, SpaceBuilder &b) {
  std::vector<IntVar> tiles;
  for (unsigned d = 0; d < in.extents.size(); ++d) {
    std::string name = prefix + "." + std::to_string(d);
    const AbstractOp::Size &extent = in.extents[d];
    tiles.push_back(extent.var ? b.divisorsOf(name, *extent.var)
                               : b.divisorsOf(name, extent.constant));
  }
  return tiles;
}

/// The footprint bound both levels post, written once: one private copy of
/// every operand tile per leaf sharing a node.
void requireFits(const AbstractOp &in, const std::vector<IntVar> &tiles,
                 IntVar tasklets, ParmValue capacity, const std::string &why,
                 SpaceBuilder &b) {
  std::vector<IntExpr> operands;
  for (const auto &dims : in.operandDims) {
    std::vector<IntExpr> factors;
    for (unsigned d : dims)
      factors.push_back(tiles[d]);
    operands.push_back(prod(factors));
  }
  b.require(tasklets * sum(operands) <= capacity, why);
}

/// `transform.cnm.distribute` -- spread the iteration space over the workgroup.
struct Distribute {
  IntVar dpus, tasklets;
  ParmValue capacity;
  unsigned level = 0;
  bool allowFloatReassociation = false;

  AbstractOp declare(const AbstractOp &in, const std::string &prefix,
                     SpaceBuilder &b) const {
    std::vector<IntVar> tiles =
        declareTiles(in, prefix + ".L" + std::to_string(level), b);

    // H4: this op splits any reduction dimension it is asked to spread, and
    // refuses to when a parallel dimension sits after it. That is a
    // precondition of the op, so the op is what has to state it -- nothing
    // downstream can, and nothing about the tile sizes implies it.
    for (unsigned d = 0; d < in.extents.size(); ++d) {
      if (!in.isReduction[d])
        continue;
      bool parallelAfter = false;
      for (unsigned e = d + 1; e < in.extents.size(); ++e)
        parallelAfter |= !in.isReduction[e];
      if (parallelAfter)
        b.require(tiles[d] == in.extents[d].expr(),
                  prefix + ".L" + std::to_string(level) + "." +
                      std::to_string(d) +
                      " must not be split (a parallel dimension follows it)");
    }

    // The same shape of precondition, read off the op's own option rather
    // than its structure: a float reduction may not be spread at all unless
    // reassociation was opted into.
    if (in.elementIsFloat && !allowFloatReassociation)
      for (unsigned d = 0; d < in.extents.size(); ++d)
        if (in.isReduction[d])
          b.require(tiles[d] == in.extents[d].expr(),
                    prefix + ".L" + std::to_string(level) + "." +
                        std::to_string(d) +
                        " must not be split (float reassociation is off)");

    // The tile counts must fill the workgroup exactly.
    std::vector<IntExpr> tilesPerDim;
    for (unsigned d = 0; d < in.extents.size(); ++d)
      tilesPerDim.push_back(in.extents[d].expr() / tiles[d]);
    b.require(prod(tilesPerDim) == dpus * tasklets,
              prefix + ": prod(extent / block) == dpus * tasklets");

    requireFits(in, tiles, tasklets, capacity,
                prefix + ": operand tiles fit the distribution's level", b);

    // Which tile dimension varies fastest across the leaves. Declared over the
    // dimensions this configuration actually spreads, which is what H3 is
    // about.
    if (in.extents.size() >= 2) {
      std::vector<BoolExpr> active;
      for (unsigned d = 0; d < in.extents.size(); ++d)
        active.push_back(in.extents[d].expr() / tiles[d] > 1);
      b.permutation(prefix + ".order", active);
    }

    // The transfer function under test: the tile becomes the extent, and
    // nothing else moves. No split is represented.
    AbstractOp out = in;
    for (unsigned d = 0; d < tiles.size(); ++d)
      out.extents[d] = {0, tiles[d]};
    return out;
  }
};

/// `transform.upmem.stage_to_wram` -- stage each tile down to the next level.
struct StageToLevel {
  IntVar tasklets;
  ParmValue capacity;
  unsigned level = 1;

  AbstractOp declare(const AbstractOp &in, const std::string &prefix,
                     SpaceBuilder &b) const {
    std::vector<IntVar> tiles =
        declareTiles(in, prefix + ".L" + std::to_string(level), b);
    requireFits(in, tiles, tasklets, capacity,
                prefix + ": operand tiles fit the staging level", b);
    AbstractOp out = in;
    for (unsigned d = 0; d < tiles.size(); ++d)
      out.extents[d] = {0, tiles[d]};
    return out;
  }
};

// ===----------------------------------------------------------------------===//
// Harness
// ===----------------------------------------------------------------------===//

using Point = std::map<std::string, ParmValue>;
using PointSet = std::set<Point>;

std::vector<int64_t> divisorsOfInt(int64_t n) {
  std::vector<int64_t> out;
  for (int64_t d = 1; d <= n; ++d)
    if (n % d == 0)
      out.push_back(d);
  return out;
}

/// Enumerate the declared box and keep what the concrete reference accepts.
/// The box is exactly what SpaceBuilder declares: a block size ranges over the
/// divisors of its extent, a leaf size over [1, extent] with divisibility left
/// to a constraint.
PointSet referenceSet(const OpDesc &op, const Platform &plat,
                      std::map<Point, unsigned> *activeCounts = nullptr) {
  const unsigned n = op.extents.size();
  const size_t levels = plat.numLevels();
  PointSet out;

  std::vector<std::vector<int64_t>> blockDomains;
  for (int64_t e : op.extents)
    blockDomains.push_back(divisorsOfInt(e));

  // The declared box, exactly as SpaceBuilder states it: the distribution's
  // tile ranges over the divisors of the extent, every inner level's over
  // [1, extent] with divisibility left to a constraint.
  std::vector<size_t> bi(n, 0);
  while (true) {
    std::vector<std::vector<int64_t>> sizes(levels, std::vector<int64_t>(n, 1));
    for (unsigned d = 0; d < n; ++d)
      sizes[0][d] = blockDomains[d][bi[d]];

    while (true) {
      for (int64_t dpus = 1; dpus <= plat.maxDpus; ++dpus)
        for (int64_t tasklets = 1; tasklets <= plat.maxTasklets; ++tasklets) {
          unsigned active = 0;
          if (!concreteAccepts(op, plat, sizes, dpus, tasklets, &active))
            continue;
          Point pt;
          for (size_t l = 0; l < levels; ++l)
            for (unsigned d = 0; d < n; ++d)
              pt["t.L" + std::to_string(l) + "." + std::to_string(d)] =
                  sizes[l][d];
          pt["dpus"] = dpus;
          pt["tasklets"] = tasklets;
          out.insert(pt);
          if (activeCounts)
            (*activeCounts)[pt] = active;
        }

      // Odometer over every inner level's tile sizes.
      size_t flat = 0;
      const size_t total = (levels - 1) * n;
      for (; flat < total; ++flat) {
        size_t l = flat / n + 1, d = flat % n;
        if (++sizes[l][d] <= op.extents[d])
          break;
        sizes[l][d] = 1;
      }
      if (flat == total)
        break;
    }

    unsigned d = 0;
    for (; d < n; ++d) {
      if (++bi[d] < blockDomains[d].size())
        break;
      bi[d] = 0;
    }
    if (d == n)
      break;
  }
  return out;
}

/// What the derived space contains, projected onto the size knobs. The
/// ordering parameter is checked separately (H3), by counting.
PointSet derivedSet(const ConfigSpace &space, unsigned numDims, size_t levels,
                    std::map<Point, size_t> *orderings = nullptr) {
  PointSet out;
  Configuration conf(space.numDims());
  std::vector<std::string> names;
  for (size_t l = 0; l < levels; ++l)
    for (unsigned d = 0; d < numDims; ++d)
      names.push_back("t.L" + std::to_string(l) + "." + std::to_string(d));
  names.push_back("dpus");
  names.push_back("tasklets");

  for (size_t i = 0; i < space.totalSize(); ++i) {
    space.at(i, conf);
    Point pt;
    for (const std::string &name : names)
      pt[name] = space.get(conf, name);
    out.insert(pt);
    if (orderings)
      (*orderings)[pt] += 1;
  }
  return out;
}

/// Build the space by folding the schedule: distribute, then stage.
void buildSchedule(const OpDesc &op, const Platform &plat, ConfigSpace &space) {
  SpaceBuilder b;
  IntVar dpus = b.intRange("dpus", 1, plat.maxDpus);
  IntVar tasklets = b.intRange("tasklets", 1, plat.maxTasklets);

  // The schedule: one distribution onto the workgroup, then one staging step
  // per memory level below it. Nothing here is written per platform -- the
  // number of levels is the length of the schedule, which is the whole point.
  AbstractOp state = initialState(op);
  state = Distribute{dpus, tasklets, static_cast<ParmValue>(plat.capacities[0]),
                     0, plat.allowFloatReassociation}
              .declare(state, "t", b);
  for (unsigned l = 1; l < plat.numLevels(); ++l)
    state =
        StageToLevel{tasklets, static_cast<ParmValue>(plat.capacities[l]), l}
            .declare(state, "t", b);

  b.buildInto(space);
}

void checkCase(const OpDesc &op, const Platform &plat) {
  SCOPED_TRACE(op.name);

  std::map<Point, unsigned> activeCounts;
  PointSet expected = referenceSet(op, plat, &activeCounts);

  ConfigSpace space;
  buildSchedule(op, plat, space);

  std::map<Point, size_t> orderings;
  PointSet actual =
      derivedSet(space, op.extents.size(), plat.numLevels(), &orderings);

  // H1 + H2 + H4: the derived space accepts exactly what the pass would.
  EXPECT_EQ(expected, actual) << "derived space disagrees with the split";

  // H3: an ordering of k active items has k! assignments, and k is what the
  // *post*-split space says it is. The derivation counted it on the pre-split
  // space, so if these agree the ordering parameter survives the split too.
  if (op.extents.size() >= 2) {
    for (const Point &pt : actual) {
      auto it = activeCounts.find(pt);
      ASSERT_NE(it, activeCounts.end());
      size_t factorial = 1;
      for (unsigned k = 2; k <= it->second; ++k)
        factorial *= k;
      EXPECT_EQ(orderings[pt], factorial)
          << "ordering multiplicity disagrees with the post-split axis count";
    }
  }

  std::fprintf(stderr, "[%s, %zu levels] reference %zu, derived %zu\n",
               op.name.c_str(), plat.numLevels(), expected.size(),
               space.totalSize());
}

// ===----------------------------------------------------------------------===//
// Cases
// ===----------------------------------------------------------------------===//

// gemv: one parallel dimension, one reduction. The reduction is the one that
// gets split, so this is the smallest case where the rank actually changes.
OpDesc gemv() {
  return {"gemv", {8, 16}, {false, true}, {{0, 1}, {1}, {0}}, 2};
}

// gemm: the reduction is last and there are two parallel dimensions before it,
// so the split is legal and prepends one dimension.
OpDesc gemm() {
  return {"gemm", {4, 4, 8}, {false, false, true}, {{0, 2}, {2, 1}, {0, 1}}, 2};
}

// Two reduction dimensions: a configuration that spreads both splits twice,
// so the iteration space grows by two and the second split runs on a space the
// first one already rewrote.
OpDesc twoReductions() {
  return {
      "two_red", {4, 4, 4}, {false, true, true}, {{0, 1, 2}, {1, 2}, {0}}, 2};
}

// A reduction followed by a parallel dimension: the pass refuses to split it.
// Nothing about the tile sizes says so -- only the op's own precondition does.
OpDesc reductionFirst() {
  return {"red_first", {8, 8}, {true, false}, {{0, 1}, {0}, {1}}, 2};
}

// No reduction at all: nothing ever splits. The control.
OpDesc allParallel() {
  return {"all_parallel", {4, 8}, {false, false}, {{0, 1}, {0, 1}}, 1};
}

TEST(SplitDerivation, Gemv) {
  checkCase(gemv(), Platform{{512, 64}, /*dpus=*/8, /*tasklets=*/4});
}

TEST(SplitDerivation, Gemm) { checkCase(gemm(), Platform{{512, 64}, 8, 4}); }

TEST(SplitDerivation, TwoReductions) {
  checkCase(twoReductions(), Platform{{512, 64}, 8, 4});
}

TEST(SplitDerivation, ReductionBeforeParallel) {
  checkCase(reductionFirst(), Platform{{512, 64}, 8, 4});
}

TEST(SplitDerivation, AllParallel) {
  checkCase(allParallel(), Platform{{512, 64}, 8, 4});
}

// The generalization the current plugin refuses outright: it errors when a
// platform declares anything other than two memory levels, because the
// pipeline has exactly two passes that consume tiling factors and a third
// level's parameters would go unread (UpmemInferAccelerator.cpp:474). Under a
// derived space the number of levels is the length of the schedule and
// nothing states it in advance -- so a three-level platform is one more
// StageToLevel in the fold, and the differential test still holds.
TEST(SplitDerivation, ThreeLevelsGemv) {
  checkCase(OpDesc{"gemv3", {4, 8}, {false, true}, {{0, 1}, {1}, {0}}, 2},
            Platform{{512, 128, 32}, 4, 4});
}

TEST(SplitDerivation, ThreeLevelsGemm) {
  checkCase(OpDesc{"gemm3",
                   {2, 2, 4},
                   {false, false, true},
                   {{0, 2}, {2, 1}, {0, 1}},
                   2},
            Platform{{512, 128, 32}, 4, 4});
}

// The gate the hand-written plugin omits. `--convert-linalg-to-cnm` is
// constructed without `allow-float-reassociation`, so it refuses to spread any
// float reduction -- but nothing in the search space says so, and every
// configuration that spreads one is proposed, lowered, and rejected at trial
// time. Reading the option off the op makes the space and the rewrite agree by
// construction.
TEST(SplitDerivation, FloatReassociationGate) {
  OpDesc intGemv = gemv();
  OpDesc floatGemv = gemv();
  floatGemv.name = "gemv_f32";
  floatGemv.elementIsFloat = true;

  Platform plat{{512, 64}, 8, 4, /*allowFloatReassociation=*/false};
  checkCase(intGemv, plat);
  checkCase(floatGemv, plat);

  ConfigSpace withGate, withoutGate;
  buildSchedule(floatGemv, plat, withGate);
  buildSchedule(intGemv, plat, withoutGate);
  std::fprintf(stderr,
               "[float gate] i32 %zu configurations, f32 %zu -- %.0f%% of the "
               "space the plugin would search on f32 cannot lower\n",
               withoutGate.totalSize(), withGate.totalSize(),
               100.0 * (1.0 - double(withGate.totalSize()) /
                                  double(withoutGate.totalSize())));
}

} // namespace
