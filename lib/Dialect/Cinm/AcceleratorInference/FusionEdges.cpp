#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/FusionEdges.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "cinm-inference"

namespace mlir::cinm {

namespace {

// ===----------------------------------------------------------------------===//
// Workgroup axis agreement
// ===----------------------------------------------------------------------===//
//
// Two ops agree on how tiles are laid out over the workgroup when they put the
// same dimension of the shared value on the same workgroup axis. Since an
// ordering is stored as one place per item (ParmKind<Permutation>), that is
// literally what it says: `producer.axis(pdim) == consumer.axis(cdim)` for the
// two iteration dimensions indexing the same dimension of the value.
//
// It used to be an opaque predicate, because a rank encoding gave no handle on
// where one item sat -- and worse, a rank ranks only the dimensions a
// configuration actually distributes, so the predicate had to re-derive that
// set per configuration before it could decode anything.

// ===----------------------------------------------------------------------===//
// Edge declaration
// ===----------------------------------------------------------------------===//
const ParmValue kUnfused = 1;

/// Position of iteration dimension `dim` among a projected permutation's
/// results, or -1 when it does not occur in them.
int inverseOf(AffineMap map, unsigned dim) {
  for (auto [position, expr] : llvm::enumerate(map.getResults()))
    if (cast<AffineDimExpr>(expr).getPosition() == dim)
      return static_cast<int>(position);
  return -1;
}

/// The place iteration dimension `dim` takes among `info`'s workgroup axes.
///
/// An op with a single iteration dimension has no order parameter -- there is
/// only one ordering of one item -- and that dimension is always the outermost
/// axis, which is place 1.
IntExpr axisOf(const DistributedOpInfo &info, unsigned dim) {
  if (info.order)
    return info.order->axis(dim);
  return IntExpr(1);
}

void requireUndistributed(const DistributedOpInfo &info, unsigned dim,
                          IntVar fuse, StringRef why, SpaceBuilder &b) {
  IntVar block = info.tiles.front()[dim];
  auto extent = static_cast<ParmValue>(info.extents[dim]);
  b.require(implies(fuse != kUnfused, block == extent),
            (fuse.name() + " fused => " + block.name() +
             " == " + std::to_string(extent) + " (" + why + ")")
                .str());
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
  IntVar fuse = b.intRange(name, kUnfused, numLevels + kUnfused);

  // The producer has to materialize whole output tiles. A dimension it does not
  // index its result with is a reduction dimension; spreading one over the
  // workgroup makes the lowering split the op and merge the partial results on
  // the host, so no leaf ever holds a finished tile.
  for (unsigned dim = 0; dim < producer.extents.size(); ++dim)
    if (inverseOf(producerMap, dim) < 0)
      requireUndistributed(producer, dim, fuse, "whole output tiles", b);

  // And the consumer must not spread a dimension the shared value is not
  // indexed by: every leaf along such an axis wants the same tile of it, so
  // the gather the producer wrote is not the one the consumer reads.
  for (unsigned dim = 0; dim < consumer.extents.size(); ++dim)
    if (inverseOf(consumerMap, dim) < 0)
      requireUndistributed(consumer, dim, fuse,
                           "every workgroup axis carries a dimension of the "
                           "shared value",
                           b);

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
      b.require(implies(fuse > level + kUnfused, producerTile == consumerTile),
                (name + " fused at level " + std::to_string(level) + " => " +
                 producerTile.name() + " == " + consumerTile.name())
                    .str());
    }

    // ...and both put it on the same workgroup axis, so that the leaf holding
    // a tile is the leaf about to read it.
    //
    // Only when the dimension is distributed: a dimension cut into one tile
    // takes no axis, and its place is then a slot in the inactive tail, which
    // the two orderings number against different item counts. Both sides are
    // distributed together or neither is -- the level-0 equality just posted
    // makes the blocks equal, and the extents already are, being the same
    // dimension of the same value -- so guarding on the producer is guarding
    // on both.
    auto producerExtent = static_cast<ParmValue>(producer.extents[producerDim]);
    BoolExpr distributed =
        producerExtent / producer.tiles.front()[producerDim] > 1;
    b.require(implies(fuse != kUnfused,
                      implies(distributed, axisOf(producer, producerDim) ==
                                               axisOf(consumer, consumerDim))),
              (name + " fused => both ops put value dimension " +
               std::to_string(position) + " on the same workgroup axis")
                  .str());
  }
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
