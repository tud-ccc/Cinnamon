#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/FusionEdges.h"
#include "cinm-mlir/Utils/Permutation.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Support/Debug.h>
#include <numeric>

#define DEBUG_TYPE "cinm-inference"

namespace mlir::cinm {

namespace {

// ===----------------------------------------------------------------------===//
// Workgroup axis model
// ===----------------------------------------------------------------------===//
//
// Two ops agree on how tiles are laid out over the workgroup when they put the
// same dimension of the shared value on the same workgroup axis. That is not an
// equality between order parameters: an order index ranks only the dimensions
// the configuration actually distributes, so what it means depends on the tile
// sizes. Hence an opaque predicate, decoding both indices per configuration.

/// Everything the axis decoding needs about one side of an edge, resolved once
/// at declaration time so the per-configuration predicate is arithmetic only.
struct AxisModel {
  SmallVector<int64_t> extents;
  /// Workgroup block size per iteration dimension.
  SmallVector<IntVar> blocks;
  /// Reduction dimensions, in op order.
  SmallVector<unsigned> reductionDims;
  /// Parallel dimensions, in op order.
  SmallVector<unsigned> parallelDims;
  /// Which dimension of the shared value each iteration dimension indexes, or
  /// -1 for one that does not index it at all.
  SmallVector<int> valueDim;
  /// Absent when the op has a single order, which is then the identity.
  std::optional<PermVar> order;
};

/// Which dimension of the shared value each workgroup axis carries, outermost
/// (slowest-varying across leaves) first. `counts[d]` is dimension `d`'s tile
/// count.
///
/// Nullopt means this configuration cannot fuse across this edge, for one of
/// two reasons: an axis carries a dimension the value is not indexed by, so
/// every leaf along that axis wants the same tile; or the order index is
/// outside the domain this configuration has, which is rejected anyway.
std::optional<SmallVector<int>> axisValueDims(const AxisModel &m,
                                              ArrayRef<int64_t> counts,
                                              int64_t orderIndex) {
  // The axes before the order index permutes them. Reduction dimensions come
  // first, in reverse op order: a reduction spread over the workgroup is split
  // during lowering into a parallel partial-sum dimension prepended to the
  // iteration space, one per split. That dimension stands for the reduction it
  // came from and indexes the same operands.
  SmallVector<unsigned> distributed;
  for (unsigned dim : llvm::reverse(m.reductionDims))
    if (counts[dim] > 1)
      distributed.push_back(dim);
  for (unsigned dim : m.parallelDims)
    if (counts[dim] > 1)
      distributed.push_back(dim);

  std::optional<int64_t> numOrders = factorial(distributed.size());
  if (!numOrders || orderIndex < 0 || orderIndex >= *numOrders)
    return std::nullopt;

  SmallVector<int> axes;
  for (unsigned position : unrankPermutation(orderIndex, distributed.size())) {
    int valueDim = m.valueDim[distributed[position]];
    if (valueDim < 0)
      return std::nullopt;
    axes.push_back(valueDim);
  }
  return axes;
}

// ===----------------------------------------------------------------------===//
// Edge declaration
// ===----------------------------------------------------------------------===//

/// Position of iteration dimension `dim` among a projected permutation's
/// results, or -1 when it does not occur in them.
int inverseOf(AffineMap map, unsigned dim) {
  for (auto [position, expr] : llvm::enumerate(map.getResults()))
    if (cast<AffineDimExpr>(expr).getPosition() == dim)
      return static_cast<int>(position);
  return -1;
}

AxisModel modelOf(const DistributedOpInfo &info, AffineMap map) {
  AxisModel model;
  model.extents = info.extents;
  model.blocks = info.tiles.front();
  model.order = info.order;
  linalg::LinalgOp op = info.op; // the interface's accessors are non-const
  SmallVector<mlir::utils::IteratorType> iteratorTypes =
      op.getIteratorTypesArray();
  for (auto [dim, kind] : llvm::enumerate(iteratorTypes))
    (kind == mlir::utils::IteratorType::parallel ? model.parallelDims
                                                 : model.reductionDims)
        .push_back(static_cast<unsigned>(dim));
  for (unsigned dim = 0; dim < info.extents.size(); ++dim)
    model.valueDim.push_back(inverseOf(map, dim));
  return model;
}

/// The workgroup layout agreement described at the top of this file, as a
/// predicate over the fused configurations.
void requireOrderAgreement(const AxisModel &producer, const AxisModel &consumer,
                           IntVar fuse, StringRef description,
                           SpaceBuilder &b) {

  // Scalar rather than vectorized. There is nothing to vectorize: the body is
  // a per-configuration decode of two ranks, so a batched form spends its
  // length hoisting rows out of the loop and indexing them back per lane, and
  // says the same thing. SpaceBuilder::require evaluates this once per
  // configuration, over the feasible set.
  b.require(
      [=](const ConfWrapper c) -> bool {
        if (fuse[c] < 2)
          return true; // unfused, so the two orders are unconstrained

        auto axesOf = [&c](const AxisModel &m) {
          SmallVector<int64_t> counts;
          for (auto [extent, block] : llvm::zip_equal(m.extents, m.blocks)) {
            // Never zero: the block sizes are divisors of the extent.
            assert(block[c] > 0 && "block size must be positive");
            counts.push_back(extent / block[c]);
          }
          // encodedValue, not get(): this reinterprets the rank against the
          // dimensions *this* configuration distributes, which is not the
          // count the parameter was declared with. The parameter is one-based
          // and the rank is zero-based, and an op with a single order has no
          // parameter at all.
          const int64_t rank = m.order ? m.order->encodedValue(c) - 1 : 0;
          return axisValueDims(m, counts, rank);
        };

        std::optional<SmallVector<int>> producerAxes = axesOf(producer);
        std::optional<SmallVector<int>> consumerAxes = axesOf(consumer);
        return producerAxes && consumerAxes && *producerAxes == *consumerAxes;
      },
      description);
}

/// Declare one edge's variable and constraints, given the maps both ops index
/// the shared value with.
void declareEdge(const DistributedOpInfo &producer,
                 const DistributedOpInfo &consumer, AffineMap producerMap,
                 AffineMap consumerMap, StringRef name, SpaceBuilder &b) {
  const ParmValue numLevels = producer.tiles.size();

  // One-based, so that no search parameter is ever zero: 1 is two launches with
  // a host round trip in between, i.e. no constraint at all, and `l + 2` means
  // the two ops additionally agree at memory level `l`.
  IntVar fuse = b.intRange(name, 1, numLevels + 1);

  // The producer has to materialize whole output tiles. A dimension it does not
  // index its result with is a reduction dimension; spreading one over the
  // workgroup makes the lowering split the op and merge the partial results on
  // the host, so no leaf ever holds a finished tile.
  for (auto [dim, extent] : llvm::enumerate(producer.extents)) {
    if (inverseOf(producerMap, dim) >= 0)
      continue;
    b.require(implies(fuse >= 2, producer.tiles.front()[dim] == extent),
              (name + " >= 2 => " + producer.tiles.front()[dim].name() +
               " == " + std::to_string(extent) + " (whole output tiles)")
                  .str());
  }

  // Per dimension of the shared value, the two ops cut it the same way: at the
  // workgroup level for any fusion at all, and at each inner level for fusion
  // that far in. This is what makes the gather and the scatter equal, and then
  // the two leaf-level loop nests fusable.
  for (unsigned position = 0; position < producerMap.getNumResults();
       ++position) {
    unsigned producerDim =
        cast<AffineDimExpr>(producerMap.getResult(position)).getPosition();
    unsigned consumerDim =
        cast<AffineDimExpr>(consumerMap.getResult(position)).getPosition();
    for (ParmValue level = 0; level < numLevels; ++level) {
      IntVar producerTile = producer.tiles[level][producerDim];
      IntVar consumerTile = consumer.tiles[level][consumerDim];
      b.require(implies(fuse >= level + 2, producerTile == consumerTile),
                (name + " >= " + std::to_string(level + 2) + " => " +
                 producerTile.name() + " == " + consumerTile.name())
                    .str());
    }
  }

  AxisModel producerModel = modelOf(producer, producerMap);
  AxisModel consumerModel = modelOf(consumer, consumerMap);
  if (producer.extents.size() > 1 || consumer.extents.size() > 1)
    requireOrderAgreement(producerModel, consumerModel, fuse,
                          (name + " >= 2 => both ops lay the same value "
                                  "dimension on the same workgroup axis")
                              .str(),
                          b);
}

} // namespace

void declareFusionEdges(ArrayRef<DistributedOpInfo> ops, SpaceBuilder &b) {
  llvm::DenseMap<Operation *, const DistributedOpInfo *> byOp;
  for (const DistributedOpInfo &info : ops)
    byOp[linalg::LinalgOp(info.op).getOperation()] = &info;

  // Two ops can be joined by more than one edge (a consumer reading the same
  // value twice, or two results of the producer); each gets its own variable.
  llvm::StringMap<unsigned> edgesPerPair;

  for (const DistributedOpInfo &producer : ops) {
    linalg::LinalgOp producerOp = producer.op;
    for (OpResult result : producerOp->getResults()) {
      AffineMap producerMap = producerOp.getIndexingMapMatchingResult(result);
      if (!producerMap.isProjectedPermutation())
        continue;
      for (OpOperand &use : result.getUses()) {
        auto it = byOp.find(use.getOwner());
        if (it == byOp.end())
          continue; // not an op the space distributes
        const DistributedOpInfo &consumer = *it->second;
        linalg::LinalgOp consumerOp = consumer.op;
        // An init operand is the consumer's destination, not something it reads
        // to compute with.
        if (consumerOp.isDpsInit(&use))
          continue;
        AffineMap consumerMap =
            consumerOp.getIndexingMapsArray()[use.getOperandNumber()];
        if (!consumerMap.isProjectedPermutation() ||
            consumerMap.getNumResults() != producerMap.getNumResults())
          continue;

        std::string name = "fuse." + producer.name + "->" + consumer.name;
        if (unsigned n = edgesPerPair[name]++)
          name += "#" + std::to_string(n);
        LLVM_DEBUG(llvm::dbgs()
                   << "[cinm-space]   fusion edge '" << name << "'\n");
        declareEdge(producer, consumer, producerMap, consumerMap, name, b);
      }
    }
  }
}

} // namespace mlir::cinm
