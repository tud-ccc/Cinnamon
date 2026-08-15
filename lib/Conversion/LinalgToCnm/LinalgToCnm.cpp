//===- LinalgToCnm.cpp - Distribute a linalg op onto a cnm workgroup -----===//
//
// Implements `--convert-linalg-to-cnm`, described in
// docs/CnmMemoryLevelsDesign.md §G.
//
// The model: treat the op as a loop nest over its iteration space. Tiling
// every dimension yields outer loops over tiles and inner loops within a
// tile. The inner loops become the `cnm.launch` body; the outer loops are
// never emitted -- they *are* the workgroup, and they determine the
// scatter/gather maps.
//
// Everything follows from one block-size vector plus, for the one thing block
// sizes cannot state, an order over the tile dimensions -- so unlike
// `--convert-cinm-to-cnm` this pass takes no decisions of its own.
//
//===----------------------------------------------------------------------===//

#include "cinm-mlir/Conversion/CnmBufferLevel.h"
#include "cinm-mlir/Conversion/CnmPasses.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmUtils.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmBase.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmOps.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmTypes.h"
#include "cinm-mlir/Utils/CinmUtils.h"
#include "cinm-mlir/Utils/Permutation.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/RegionUtils.h>

#include <numeric>

namespace mlir {
#define GEN_PASS_DEF_CONVERTLINALGTOCNMPASS
#include <cinm-mlir/Conversion/CnmPasses.h.inc>
} // namespace mlir

using namespace mlir;

namespace {

/// How one operand is spread over the workgroup.
struct OperandTiling {
  /// Extent of one tile along each of the operand's own dimensions. This is
  /// the `cnm.buffer` shape.
  SmallVector<int64_t> blocks;
  /// (workgroup coordinates, buffer coordinates) -> host element.
  AffineMap scatterMap;
};

/// The pass options `distribute` reads, in one struct so that adding an option
/// does not mean threading another argument through every helper.
struct DistributionOptions {
  StringRef bufferLevel;
  StringRef leafTileAttr;
  bool allowFloatReassociation;
  ArrayRef<std::string> perDimAttrs;
  ArrayRef<int64_t> workgroupDimOrder;
  int64_t workgroupDimOrderIndex;
};

//===----------------------------------------------------------------------===//
// Distribution
//===----------------------------------------------------------------------===//

/// Clone `op` with new operands, dropping the results. Used to place the op
/// into the launch body, where it computes on leaf-sized memrefs.
///
/// Everything except the tile sizes carries over unchanged, which is what we
/// want: `indexing_maps`, `dimensions`, `permutation` and friends are stated
/// in terms of iteration-space dimensions, and tiling preserves those. The
/// inherent attributes travel as properties (so `operandSegmentSizes` and the
/// like stay out of the discardable dictionary); the operand counts are the
/// same, so they need no adjustment.
static Operation *cloneOnBuffers(ImplicitLocOpBuilder &b, linalg::LinalgOp op,
                                 ValueRange ins, ValueRange outs) {
  OperationState state(b.getLoc(), op->getName());
  state.addOperands(ins);
  state.addOperands(outs);
  state.propertiesAttr = op->getPropertiesAsAttribute();
  for (NamedAttribute attr : op->getDiscardableAttrs())
    if (attr.getName() != cnm::CnmDialect::TILE_SIZES_NAME &&
        attr.getName() != cnm::CnmDialect::WORKGROUP_DIM_ORDER_NAME &&
        attr.getName() != cnm::CnmDialect::WORKGROUP_DIM_ORDER_INDEX_NAME)
      state.addAttribute(attr.getName(), attr.getValue());
  for (Region &region : op->getRegions()) {
    auto cloned = std::make_unique<Region>();
    IRMapping mapping;
    region.cloneInto(cloned.get(), mapping);
    state.addRegion(std::move(cloned));
  }
  return b.create(state);
}

/// How each iteration dimension is cut for the tile the body will later stage
/// into leaf memory, and where the pieces land in the split iteration space.
///
/// A leaf's buffer laid out in operand order is generally strided when sliced
/// by a staged tile: for an M x K tile staged K-chunk at a time, the chunk's M
/// rows sit K elements apart. Cutting the dimension in two and putting the
/// chunk dimension outermost makes that slice one contiguous run, which is the
/// only thing a DMA moves. The iteration space has to be cut the same way, or
/// the body would no longer index the buffer it was given.
struct LeafSplit {
  /// Per original dimension, how many staged chunks it is cut into; 1 means
  /// the whole dimension is staged at once and it is not split.
  SmallVector<int64_t> chunks;
  /// Per original dimension, the extent of one staged tile.
  SmallVector<int64_t> tiles;
  /// Per original dimension, its pieces' positions in the split space.
  /// `chunkDim` is -1 for a dimension that was not split.
  SmallVector<int64_t> chunkDim, tileDim;
  unsigned numDims = 0;

  bool splits() const {
    return llvm::any_of(chunks, [](int64_t c) { return c > 1; });
  }
};

/// Reads the staged tile from `op` and works out the resulting split. A
/// dimension is split only when its block is a proper multiple of its tile:
/// equal means the whole block is staged at once, and a tile that does not
/// divide the block describes no regular chunking.
static LeafSplit computeLeafSplit(linalg::LinalgOp op, ArrayRef<int64_t> blocks,
                                  const DistributionOptions &options) {
  unsigned numLoops = op.getNumLoops();
  LeafSplit split;
  split.chunks.assign(numLoops, 1);
  split.tiles.assign(blocks.begin(), blocks.end());
  split.chunkDim.assign(numLoops, -1);
  split.tileDim.assign(numLoops, 0);

  ArrayRef<int64_t> leaf;
  DenseI64ArrayAttr attr;
  if (!options.leafTileAttr.empty())
    attr = op->getAttrOfType<DenseI64ArrayAttr>(options.leafTileAttr);
  if (attr && attr.size() == static_cast<int64_t>(numLoops))
    leaf = attr.asArrayRef();

  for (unsigned d = 0; d < numLoops; ++d)
    if (!leaf.empty() && leaf[d] > 0 && leaf[d] < blocks[d] &&
        blocks[d] % leaf[d] == 0) {
      split.chunks[d] = blocks[d] / leaf[d];
      split.tiles[d] = leaf[d];
    }

  // A dimension's chunk sits immediately before its tile, so the split space
  // reads as the original one with dimensions expanded in place.
  unsigned next = 0;
  for (unsigned d = 0; d < numLoops; ++d) {
    if (split.chunks[d] > 1)
      split.chunkDim[d] = next++;
    split.tileDim[d] = next++;
  }
  split.numDims = next;
  return split;
}

/// Every indexing map must be a projected permutation: the tiling of an
/// operand dimension is then simply the tiling of the one loop dimension that
/// indexes it. Anything else (a repeated dimension, a window) would need the
/// tile to be a strided or overlapping slice, which `cnm.buffer` cannot
/// describe. Dropping a dimension is fine -- that is a broadcast operand.
static LogicalResult checkProjectedPermutations(linalg::LinalgOp op) {
  for (auto [operand, map] :
       llvm::zip(op->getOpOperands(), op.getIndexingMapsArray()))
    if (!map.isProjectedPermutation())
      return op->emitOpError("cannot distribute operand #")
             << operand.getOperandNumber() << ": indexing map "
             << AffineMapAttr::get(map) << " is not a projected permutation";
  return success();
}

/// Loop extents, read off the operands. Only valid once
/// `checkProjectedPermutations` has passed.
static FailureOr<SmallVector<int64_t>> getLoopExtents(linalg::LinalgOp op) {
  SmallVector<int64_t> extents(op.getNumLoops(), ShapedType::kDynamic);
  for (auto [operand, map] :
       llvm::zip(op->getOpOperands(), op.getIndexingMapsArray())) {
    auto shape = asShaped(operand.get().getType()).getShape();
    for (auto [pos, expr] : llvm::enumerate(map.getResults()))
      extents[cast<AffineDimExpr>(expr).getPosition()] = shape[pos];
  }
  for (auto [dim, extent] : llvm::enumerate(extents))
    if (ShapedType::isDynamic(extent))
      return op->emitOpError("iteration dimension ")
             << dim
             << " has a dynamic extent; the distribution is computed "
                "from static sizes";
  return extents;
}

/// Tile counts `extent / block`, checking that the block sizes make sense for
/// this op.
static FailureOr<SmallVector<int64_t>>
getTileCounts(linalg::LinalgOp op, ArrayRef<int64_t> blocks,
              ArrayRef<int64_t> extents) {
  if (blocks.size() != op.getNumLoops())
    return op->emitOpError("expected ")
           << op.getNumLoops() << " block size(s) in '"
           << cnm::CnmDialect::TILE_SIZES_NAME << "' (one per iteration "
           << "dimension), got " << blocks.size();

  SmallVector<int64_t> counts(blocks.size());
  for (auto [dim, extent, block] : llvm::enumerate(extents, blocks)) {
    if (block <= 0)
      return op->emitOpError("block size for iteration dimension ")
             << dim << " must be positive, got " << block;
    if (extent % block != 0)
      return op->emitOpError("block size ")
             << block << " does not divide the extent " << extent
             << " of iteration dimension " << dim;
    counts[dim] = extent / block;
  }
  return counts;
}

/// Reshapes the host-side merge `splitReduction` just emitted.
///
/// It comes out with the partial-sum dimension outermost, so the merge sweeps
/// the whole output once per leaf, reloading and restoring every element each
/// time. Running the reduction innermost instead keeps one output element live
/// across its whole sum, which is what lets --affine-scalrep hold it in a
/// register and a vectorizer widen the parallel dimension rather than the
/// accumulation.
///
/// Its `outs` is the original op's, which sits where the frontend put it --
/// before everything this pass emits in between. Sinking it to its only
/// consumer gives loop fusion an adjacent pair, and stops a buffer that is
/// dead until the end from living across the whole device section.
static LogicalResult shapeHostMerge(RewriterBase &rewriter,
                                    linalg::SplitReductionResult split) {
  linalg::LinalgOp mergeOp = split.resultCombiningLinalgOp;
  auto merge = dyn_cast<linalg::GenericOp>(mergeOp.getOperation());
  if (!merge)
    return success();

  // Parallel dimensions first, keeping their order, then the reduction ones.
  // `interchange` is the permutation to read the current dimensions in.
  SmallVector<unsigned> permutation;
  for (utils::IteratorType wanted :
       {utils::IteratorType::parallel, utils::IteratorType::reduction})
    for (auto [dim, kind] : llvm::enumerate(merge.getIteratorTypesArray()))
      if (kind == wanted)
        permutation.push_back(dim);

  if (!llvm::is_sorted(permutation)) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(merge);
    FailureOr<linalg::GenericOp> interchanged =
        linalg::interchangeGenericOp(rewriter, merge, permutation);
    if (failed(interchanged))
      return merge->emitOpError(
          "could not put the host-side merge's reduction innermost");
    merge = *interchanged;
  }

  auto *initOpnd = merge.getDpsInitOperand(0);
  if (Operation *init = initOpnd->get().getDefiningOp()) {
    if (init->getBlock() == merge->getBlock() && init->isBeforeInBlock(merge) &&
        llvm::hasSingleElement(init->getUsers()))
      rewriter.moveOpBefore(init, merge);
  }

  return success();
}

/// Spread across the workgroup every reduction dimension whose block size asks
/// for it, by rewriting the op into a partial-reduction op plus a host-side
/// merge (design §G4).
///
/// Upstream's `splitReduction` performs exactly this rewrite, and it also
/// covers §G5: it seeds the partial result with the combiner's neutral
/// element, so every leaf starts from the reduction identity, and its merge op
/// accumulates into the *original* `outs`, so an incoming accumulator is
/// folded in exactly once rather than once per leaf.
///
/// The split dimension is inserted *before* the parallel dimensions, so under
/// the default order of `getWorkgroupAxisOrder` it is the outermost one and the
/// original parallel dimensions vary fastest across leaves. The leaves sharing
/// one node of the workgroup (tasklets within a DPU, on UPMEM) then differ in
/// their parallel tile and share their reduction tile, which is what lets a
/// broadcast operand indexed only by reduction dimensions -- gemv's vector --
/// be stored once per node instead of replicated per leaf. See §G3 for the
/// evidence and for what it costs; the order is a parameter, so the opposite
/// trade is reachable without touching this rewrite.
///
/// `blocks` is updated to describe the rewritten op.
static FailureOr<linalg::LinalgOp>
splitDistributedReductions(RewriterBase &rewriter, linalg::LinalgOp op,
                           SmallVector<int64_t> &blocks,
                           const DistributionOptions &options) {
  while (true) {
    FailureOr<SmallVector<int64_t>> extents = getLoopExtents(op);
    if (failed(extents))
      return failure();

    auto iterators = op.getIteratorTypesArray();
    std::optional<unsigned> target;
    for (auto [dim, kind] : llvm::enumerate(iterators)) {
      if (kind == utils::IteratorType::parallel) {
        // The insert position below assumes the parallel dimensions come
        // first, which also makes `blocks` line up with the rewritten op.
        if (target)
          return op->emitOpError(
              "cannot split a reduction dimension of an op whose parallel "
              "dimensions do not all come first");
        continue;
      }
      if (!target && (*extents)[dim] != blocks[dim])
        target = dim;
    }
    if (!target)
      return op;

    int64_t ratio = (*extents)[*target] / blocks[*target];

    // Reassociating a float reduction changes the result, so it is opt-in
    // rather than something the search does behind the user's back (§G6).
    Type elementType = asShaped(op.getDpsInits()[0].getType()).getElementType();
    if (isa<FloatType>(elementType) && !options.allowFloatReassociation)
      return op->emitOpError("splitting reduction dimension ")
             << *target << " " << ratio
             << " ways reassociates a floating-point reduction, which changes "
                "the result; pass allow-float-reassociation to permit it";

    linalg::ControlSplitReductionFn control = [&](linalg::LinalgOp) {
      return linalg::SplitReductionOptions{ratio, /*index=*/0,
                                           /*innerParallel=*/false};
    };
    FailureOr<linalg::SplitReductionResult> split =
        linalg::splitReduction(rewriter, op, control);
    if (failed(split))
      return op->emitOpError("could not split reduction dimension ")
             << *target
             << ": its combiner was not recognised as one with a neutral "
                "element";

    if (failed(shapeHostMerge(rewriter, *split)))
      return failure();

    // The rewritten iteration space is [split dim] ++ [parallel dims] ++
    // [reduction dims], with the split dimension holding one tile per leaf and
    // the original reduction dimension now spanning exactly one block.
    blocks.insert(blocks.begin(), 1);

    // splitReduction builds a fresh op, so whatever the pipeline stamped on
    // this one has to be carried across -- otherwise a decision made upstream
    // silently disappears exactly when a reduction is split. Lists indexed by
    // iteration dimension additionally get an entry for the new dimension.
    unsigned oldNumLoops = op.getNumLoops();
    DictionaryAttr carried = op->getDiscardableAttrDictionary();
    op = split->splitLinalgOp;
    op->setDiscardableAttrs(carried);
    for (StringRef name : llvm::concat<const std::string>(
             SmallVector<std::string>{cnm::CnmDialect::TILE_SIZES_NAME.str()},
             options.perDimAttrs)) {
      auto attr = op->getAttrOfType<DenseI64ArrayAttr>(name);
      if (!attr || attr.size() != static_cast<int64_t>(oldNumLoops))
        continue;
      SmallVector<int64_t> updated(attr.asArrayRef());
      updated.insert(updated.begin(), 1);
      op->setAttr(name, DenseI64ArrayAttr::get(op->getContext(), updated));
    }

    // An order is a permutation of the iteration dimensions, so it does not
    // get an entry inserted like the size lists above -- it gets rewritten.
    // Every old dimension has moved up by one; the prepended partial-sum
    // dimension stands for the reduction that was split, so it takes that
    // reduction's axis; and the reduction itself now spans a single block, so
    // it takes no axis at all and goes last.
    //
    // This is what lets the order arrive as an explicit permutation over the
    // dimensions the op had *before* this pass ran. Whoever stamps it does not
    // have to predict the split, because the split fixes it up -- which is the
    // alternative to a rank, whose whole reason for existing was to be
    // invariant to a rewrite it could not see.
    StringRef orderName = cnm::CnmDialect::WORKGROUP_DIM_ORDER_NAME;
    if (auto attr = op->getAttrOfType<DenseI64ArrayAttr>(orderName)) {
      if (attr.size() == static_cast<int64_t>(oldNumLoops)) {
        auto splitDim = static_cast<int64_t>(*target);
        SmallVector<int64_t> updated;
        updated.reserve(oldNumLoops + 1);
        for (int64_t dim : attr.asArrayRef())
          updated.push_back(dim == splitDim ? 0 : dim + 1);
        updated.push_back(splitDim + 1);
        op->setAttr(orderName,
                    DenseI64ArrayAttr::get(op->getContext(), updated));
      }
    }

    FailureOr<SmallVector<int64_t>> newExtents = getLoopExtents(op);
    if (failed(newExtents))
      return failure();
    if ((*newExtents)[0] != ratio)
      return op->emitOpError("internal error: splitReduction placed the split "
                             "dimension somewhere unexpected");
  }
}

//===----------------------------------------------------------------------===//
// Tile-dimension -> workgroup-axis order
//===----------------------------------------------------------------------===//

/// The order asked for, and where it was asked for. Both forms are empty when
/// nothing was asked for, in which case the default rule applies.
struct OrderRequest {
  /// A permutation of the op's iteration dimensions.
  ArrayRef<int64_t> permutation;
  /// The same choice by lexicographic rank; negative means unset.
  int64_t index = -1;
  /// How each was spelled, so a diagnostic can point at what to change.
  StringRef permutationName, indexName;
};

/// Where the order comes from: the op's own attributes if it carries any,
/// otherwise the pass options.
///
/// Attributes win rather than conflict. The block sizes already arrive per op
/// as `cnm.tile_sizes`, stamped by the search; the order is stamped the same
/// way, and a pass option is then a default for the ops the search said
/// nothing about.
static FailureOr<OrderRequest>
getOrderRequest(linalg::LinalgOp op, const DistributionOptions &options) {
  StringRef permutationAttrName = cnm::CnmDialect::WORKGROUP_DIM_ORDER_NAME;
  StringRef indexAttrName = cnm::CnmDialect::WORKGROUP_DIM_ORDER_INDEX_NAME;
  auto permutationAttr =
      op->getAttrOfType<DenseI64ArrayAttr>(permutationAttrName);
  auto indexAttr = op->getAttrOfType<IntegerAttr>(indexAttrName);

  if (op->hasAttr(permutationAttrName) && !permutationAttr)
    return op->emitOpError("'")
           << permutationAttrName << "' must be a dense i64 array giving a "
           << "permutation of the iteration dimensions";
  if (op->hasAttr(indexAttrName) && !indexAttr)
    return op->emitOpError("'")
           << indexAttrName << "' must be an integer attribute";

  OrderRequest request;
  if (permutationAttr || indexAttr) {
    if (permutationAttr && indexAttr)
      return op->emitOpError("carries both '")
             << permutationAttrName << "' and '" << indexAttrName
             << "', which are two ways of stating the same thing; keep one";
    request.permutationName = permutationAttrName;
    request.indexName = indexAttrName;
    if (permutationAttr)
      request.permutation = permutationAttr.asArrayRef();
    if (indexAttr)
      request.index = indexAttr.getInt();
    return request;
  }

  if (!options.workgroupDimOrder.empty() && options.workgroupDimOrderIndex >= 0)
    return op->emitOpError(
        "'workgroup-dim-order' and 'workgroup-dim-order-index' are two ways of "
        "stating the same thing; pass at most one");
  request.permutationName = "workgroup-dim-order";
  request.indexName = "workgroup-dim-order-index";
  request.permutation = options.workgroupDimOrder;
  request.index = options.workgroupDimOrderIndex;
  return request;
}

/// Which iteration dimension occupies which workgroup axis, outermost (that
/// is, slowest-varying across leaves) first.
///
/// The default is the rule of design §G3: parallel dimensions outer, reduction
/// dimensions inner, op order within each group. It is only a default now --
/// which dimension varies fastest decides which operands the leaves sharing a
/// hardware node replicate rather than share, and that is worth searching.
///
/// Two ways to override it, because two callers want different things:
///
/// - a permutation stated outright, over all iteration dimensions the op has
///   *here*, i.e. after any reduction split has prepended one. Readable, and
///   the right form for a lit test.
/// - the `index`-th order in lexicographic order, so the parameter is an
///   integer a search can enumerate. This is what the search stamps.
///
/// The index deliberately ranks over *only* the dimensions whose tile count is
/// greater than one. A dimension tiled once occupies no workgroup axis -- its
/// tile coordinate is the constant 0 -- so moving it changes nothing, and
/// ranking over all dimensions would hand a search a space that is mostly
/// duplicates (a split gemv: 6 orders, 2 of them distinct). Note that by the
/// time we get here every reduction spread across the workgroup has been split
/// away, so the reduction dimensions that remain always have count 1: what the
/// index permutes is exactly the distributed parallel dimensions, in op order.
/// Index 0 is therefore the identity *and* the rule above.
static FailureOr<SmallVector<unsigned>>
getWorkgroupAxisOrder(linalg::LinalgOp op, ArrayRef<int64_t> counts,
                      const DistributionOptions &options) {
  auto numLoops = static_cast<int64_t>(op.getNumLoops());

  FailureOr<OrderRequest> request = getOrderRequest(op, options);
  if (failed(request))
    return failure();

  if (!request->permutation.empty()) {
    if (static_cast<int64_t>(request->permutation.size()) != numLoops)
      return op->emitOpError("'")
             << request->permutationName << "' has "
             << request->permutation.size() << " entries but this op has "
             << numLoops
             << " iteration dimension(s) (splitting a reduction across the "
                "workgroup prepends one, so this is the count *after* the "
                "split, not the one the source op had)";

    SmallVector<unsigned> order;
    SmallVector<bool> seen(numLoops, false);
    for (int64_t dim : request->permutation) {
      if (dim < 0 || dim >= numLoops || seen[dim])
        return op->emitOpError("'")
               << request->permutationName << "' must be a permutation of [0, "
               << numLoops << "), but " << dim
               << " is out of range or repeated";
      seen[dim] = true;
      order.push_back(dim);
    }
    return order;
  }

  auto iteratorTypes = op.getIteratorTypesArray();
  SmallVector<unsigned> order;
  for (auto [dim, kind] : llvm::enumerate(iteratorTypes))
    if (kind == utils::IteratorType::parallel)
      order.push_back(dim);
  for (auto [dim, kind] : llvm::enumerate(iteratorTypes))
    if (kind != utils::IteratorType::parallel)
      order.push_back(dim);

  if (request->index < 0)
    return order;

  SmallVector<unsigned> distributed, single;
  for (unsigned dim : order)
    (counts[dim] == 1 ? single : distributed).push_back(dim);

  std::optional<int64_t> numOrders = cinm::factorial(distributed.size());
  if (!numOrders)
    return op->emitOpError("spreads ")
           << distributed.size()
           << " iteration dimensions over the workgroup, too many to index "
              "their orders";
  if (request->index >= *numOrders)
    return op->emitOpError("'")
           << request->indexName << "' is " << request->index
           << ", but this op spreads " << distributed.size()
           << " iteration dimension(s) over the workgroup, so it has "
           << *numOrders << " distinct order(s)";

  SmallVector<unsigned> ranked;
  for (unsigned position :
       cinm::unrankPermutation(request->index, distributed.size()))
    ranked.push_back(distributed[position]);
  // The dimensions tiled once take no workgroup axis, so they can go anywhere;
  // keeping them last keeps the printed order readable.
  llvm::append_range(ranked, single);
  return ranked;
}

LogicalResult distribute(RewriterBase &rewriter, linalg::LinalgOp op,
                         const DistributionOptions &options) {
  auto tileAttr =
      op->getAttrOfType<DenseI64ArrayAttr>(cnm::CnmDialect::TILE_SIZES_NAME);
  assert(tileAttr && "caller filters on the attribute");

  if (!op.hasPureTensorSemantics())
    return op->emitOpError("can only be distributed onto a workgroup while it "
                           "still has pure tensor semantics; run this pass "
                           "before bufferization");

  auto accelerator =
      cinm::getEnclosingAcceleratorAs<cnm::CnmAcceleratorAttrInterface>(op);
  if (!accelerator)
    return op->emitOpError("is not inside a compute block with an accelerator, "
                           "so there is no workgroup to distribute onto");

  FailureOr<cnm::BufferLevel> level =
      cnm::resolveBufferLevel(options.bufferLevel, accelerator, op);
  if (failed(level))
    return failure();

  if (failed(checkProjectedPermutations(op)))
    return failure();

  SmallVector<int64_t> blocks(tileAttr.asArrayRef());
  {
    FailureOr<SmallVector<int64_t>> extents = getLoopExtents(op);
    if (failed(extents))
      return failure();
    FailureOr<SmallVector<int64_t>> counts =
        getTileCounts(op, blocks, *extents);
    if (failed(counts))
      return failure();

    ArrayRef<int64_t> wgShape = accelerator.getWorkgroupShape();
    int64_t numLeaves = std::reduce(wgShape.begin(), wgShape.end(), int64_t{1},
                                    std::multiplies<>());
    int64_t numTiles = std::reduce(counts->begin(), counts->end(), int64_t{1},
                                   std::multiplies<>());
    if (numTiles != numLeaves)
      return op->emitOpError("the block sizes produce ")
             << numTiles << " tile(s) but the workgroup has " << numLeaves
             << " leaves; their product must match exactly (sequential trips "
                "over the problem belong to a tiling pass upstream)";
  }

  // Any reduction dimension the block sizes spread over the workgroup becomes
  // a parallel dimension over partial results, plus a merge left on the host.
  // Everything below therefore only ever sees unsplit reductions.
  FailureOr<linalg::LinalgOp> split =
      splitDistributedReductions(rewriter, op, blocks, options);
  if (failed(split))
    return failure();
  op = *split;

  LeafSplit leafSplit = computeLeafSplit(op, blocks, options);
  // Laying a buffer out for the tile its body stages restates the iteration
  // space, and a named op's maps and iterator kinds are implied by its name,
  // so there is nowhere to state it.
  if (leafSplit.splits() && !isa<linalg::GenericOp>(op.getOperation()))
    return op->emitOpError(
        "is not linalg.generic, "
        "run --linalg-generalize-named-ops before this pass");

  auto indexingMaps = op.getIndexingMapsArray();
  unsigned numLoops = op.getNumLoops();

  FailureOr<SmallVector<int64_t>> loopExtents = getLoopExtents(op);
  if (failed(loopExtents))
    return failure();
  FailureOr<SmallVector<int64_t>> tileCounts =
      getTileCounts(op, blocks, *loopExtents);
  if (failed(tileCounts))
    return failure();
  SmallVector<int64_t> counts = *tileCounts;
  ArrayRef<int64_t> wgShape = accelerator.getWorkgroupShape();

  // Tile-space -> workgroup mapping. Both sides are linearized; the tile-side
  // order defaults to the rule of design §G3 and is otherwise whatever the
  // options ask for. It is a real choice: the dimension that varies fastest
  // across leaves is the one the leaves sharing a hardware node differ in.
  FailureOr<SmallVector<unsigned>> order =
      getWorkgroupAxisOrder(op, counts, options);
  if (failed(order))
    return failure();

  SmallVector<int64_t> strides(numLoops);
  int64_t stride = 1;
  for (unsigned dim : llvm::reverse(*order)) {
    strides[dim] = stride;
    stride *= counts[dim];
  }

  MLIRContext *ctx = op->getContext();
  AffineExpr leaf = linearizeIndices(ctx, wgShape);
  SmallVector<AffineExpr> tileCoords(numLoops);
  for (unsigned dim = 0; dim < numLoops; ++dim)
    tileCoords[dim] = counts[dim] == 1
                          ? getAffineConstantExpr(0, ctx)
                          : leaf.floorDiv(strides[dim]) % counts[dim];

  // Per-operand tiling, derived from the indexing maps. The scatter map names
  // a host element for every element of every leaf's buffer.
  //
  // Without a split the operand is scattered as it stands, in whatever layout
  // it already has. With one, each of its dimensions that gets staged in
  // chunks contributes two buffer dimensions, and all the chunk dimensions are
  // placed before all the tile dimensions -- that ordering is the whole point,
  // since it is what makes one staged chunk a contiguous run. The scatter map
  // absorbs the reordering, so the host value is still read where it lies.
  SmallVector<OperandTiling> tilings;
  SmallVector<AffineMap> bodyMaps;
  for (auto [operand, map] : llvm::zip(op->getOpOperands(), indexingMaps)) {
    OperandTiling tiling;
    SmallVector<AffineExpr> scatterResults;
    SmallVector<AffineExpr> bodyResults;

    // Chunk dimensions first, then tile dimensions, each in operand order.
    SmallVector<unsigned> chunkPositions, tilePositions;
    for (auto [position, expr] : llvm::enumerate(map.getResults())) {
      unsigned dim = cast<AffineDimExpr>(expr).getPosition();
      if (leafSplit.chunks[dim] > 1)
        chunkPositions.push_back(position);
      tilePositions.push_back(position);
    }
    auto dimOf = [&](unsigned position) {
      return cast<AffineDimExpr>(map.getResult(position)).getPosition();
    };
    for (unsigned position : chunkPositions) {
      tiling.blocks.push_back(leafSplit.chunks[dimOf(position)]);
      bodyResults.push_back(
          getAffineDimExpr(leafSplit.chunkDim[dimOf(position)], ctx));
    }
    for (unsigned position : tilePositions) {
      tiling.blocks.push_back(leafSplit.tiles[dimOf(position)]);
      bodyResults.push_back(
          getAffineDimExpr(leafSplit.tileDim[dimOf(position)], ctx));
    }

    // The host element for a buffer element: which tile the leaf holds, then
    // which chunk of it, then where in the chunk.
    unsigned numChunkDims = chunkPositions.size();
    for (auto [index, position] : llvm::enumerate(tilePositions)) {
      unsigned dim = dimOf(position);
      AffineExpr within =
          getAffineDimExpr(wgShape.size() + numChunkDims + index, ctx);
      if (leafSplit.chunks[dim] > 1) {
        unsigned chunkIndex =
            llvm::find(chunkPositions, position) - chunkPositions.begin();
        within = getAffineDimExpr(wgShape.size() + chunkIndex, ctx) *
                     leafSplit.tiles[dim] +
                 within;
      }
      scatterResults.push_back(tileCoords[dim] * blocks[dim] + within);
    }

    SmallVector<int64_t> bounds(wgShape);
    llvm::append_range(bounds, tiling.blocks);
    tiling.scatterMap = simplifyAffineMapWithBounds(
        AffineMap::get(wgShape.size() + tiling.blocks.size(), 0, scatterResults,
                       ctx),
        bounds);
    tilings.push_back(std::move(tiling));
    bodyMaps.push_back(AffineMap::get(leafSplit.numDims, 0, bodyResults, ctx));
  }

  // The body iterates the split space; a split dimension's pieces inherit its
  // kind, so a chunked reduction stays two reductions.
  SmallVector<utils::IteratorType> bodyIterators(leafSplit.numDims);
  for (unsigned d = 0, e = op.getNumLoops(); d < e; ++d) {
    utils::IteratorType kind = op.getIteratorTypesArray()[d];
    if (leafSplit.chunkDim[d] >= 0)
      bodyIterators[leafSplit.chunkDim[d]] = kind;
    bodyIterators[leafSplit.tileDim[d]] = kind;
  }

  //===--------------------------------------------------------------------===//
  // Emit
  //===--------------------------------------------------------------------===//

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);

  auto workgroup =
      cnm::WorkgroupOp::create(b, accelerator.getWorkgroupType()).getResult();

  unsigned numInputs = op.getDpsInputs().size();
  SmallVector<Value> launchInputs, launchOutputs;
  for (auto [i, operand, tiling] :
       llvm::enumerate(op->getOpOperands(), tilings)) {
    auto operandTy = asShaped(operand.get().getType());
    auto bufferTy = cnm::BufferType::get(
        tiling.blocks, operandTy.getElementType(), accelerator, level->space);
    Value alloc = cnm::DeclareBufferOp::create(b, bufferTy, workgroup);
    bool isDestination = operand.getOperandNumber() >= numInputs;

    if (isDestination &&
        isa_and_nonnull<tensor::EmptyOp>(operand.get().getDefiningOp())) {
      // A freshly allocated destination has undefined contents, so there is
      // nothing to bring over.
    } else {
      // cnm.scatter moves shaped values, and an operand no iteration
      // dimension indexes need not be one: fusion pulls a loop-invariant
      // scalar in as an operand with an empty indexing map, which is what
      // geva's two coefficients become. Its buffer is already the rank-0 one
      // that scalar stands for, so materializing the tensor to match puts it
      // on the same path as every other operand -- and the right one, since a
      // value every leaf needs whole is exactly what the scatter below is
      // then optimized into a broadcast.
      Value source = operand.get();
      if (!isa<ShapedType>(source.getType()))
        source = tensor::FromElementsOp::create(b, operandTy, source);

      // This scatter can then be optimized into a broadcast.
      // For now this happens in the backend dialect, which has knowledge of
      // the constraints on the scatter calls.
      auto scatter = cnm::ScatterOp::create(b, source, alloc, workgroup,
                                            tiling.scatterMap);
      scatter->setAttr(cinm::CinmDialect::DEBUG_TAG_NAME,
                       b.getStringAttr(cinm::isStaticValue(operand.get())
                                           ? "static"
                                           : "dyn"));
    }

    (isDestination ? launchOutputs : launchInputs).push_back(alloc);
  }

  auto launch =
      cnm::LaunchOp::create(b, workgroup, launchInputs, launchOutputs);
  {
    Block &body = launch.getBody().emplaceBlock();
    for (Value param : launch.getParams()) {
      auto bufferTy = cast<cnm::BufferType>(param.getType());
      // The buffer's level becomes the memref's memory space; that is what
      // LaunchOp::verify requires, and it is how the body learns which
      // memory it computes on.
      body.addArgument(
          MemRefType::get(bufferTy.getShape(), bufferTy.getElementType(),
                          MemRefLayoutAttrInterface{}, bufferTy.getLevel()),
          param.getLoc());
    }
    OpBuilder::InsertionGuard bodyGuard(b);
    b.setInsertionPointToStart(&body);
    ValueRange bodyIns = body.getArguments().take_front(numInputs);
    ValueRange bodyOuts = body.getArguments().drop_front(numInputs);
    if (leafSplit.splits()) {
      // Clone exactly as the unsplit path does -- that is what gets the body
      // region and its attributes right -- then restate the maps and iterator
      // kinds over the split space the buffers were reshaped into.
      Operation *cloned = cloneOnBuffers(b, op, bodyIns, bodyOuts);
      auto generic = cast<linalg::GenericOp>(cloned);
      generic.setIndexingMapsAttr(b.getAffineMapArrayAttr(bodyMaps));
      generic.setIteratorTypesAttr(b.getArrayAttr(llvm::map_to_vector(
          bodyIterators, [&](utils::IteratorType kind) -> Attribute {
            return linalg::IteratorTypeAttr::get(b.getContext(), kind);
          })));
      // The staging pass tiles the split space: one chunk at a time along the
      // dimensions that were cut, the whole extent along the rest.
      if (!options.leafTileAttr.empty()) {
        SmallVector<int64_t> staged(leafSplit.numDims);
        for (unsigned d = 0, e = leafSplit.chunks.size(); d < e; ++d) {
          if (leafSplit.chunkDim[d] >= 0)
            staged[leafSplit.chunkDim[d]] = 1;
          staged[leafSplit.tileDim[d]] = leafSplit.tiles[d];
        }
        generic->setAttr(options.leafTileAttr, b.getDenseI64ArrayAttr(staged));
      }
    } else {
      cloneOnBuffers(b, op, bodyIns, bodyOuts);
    }
    cnm::ReturnOp::create(b);
  }

  // Gather each result straight back into a value of the op's own shape.
  SmallVector<Value> results;
  for (auto [index, init] : llvm::enumerate(op.getDpsInits())) {
    const OperandTiling &tiling = tilings[numInputs + index];
    auto initTy = asShaped(init.getType());
    Value destination =
        tensor::EmptyOp::create(b, initTy.getShape(), initTy.getElementType());
    results.push_back(cnm::GatherOp::create(b, launchOutputs[index], workgroup,
                                            tiling.scatterMap, destination)
                          .getOutput());
  }

  cnm::FreeWorkgroupOp::create(b, workgroup);
  rewriter.replaceOp(op, results);
  return success();
}

static bool isIsolatedFromAbove(Region &region) {
  bool result = true;
  mlir::visitUsedValuesDefinedAbove({region}, [&](auto *) { result = false; });
  return result;
}

struct ConvertLinalgToCnmPass
    : public impl::ConvertLinalgToCnmPassBase<ConvertLinalgToCnmPass> {
  using Base::Base;

  void runOnOperation() final {
    // Collect first: distributing an op rewrites the region around it, and
    // the ops we create inside `cnm.launch` are deliberately not candidates
    // (they have no tile sizes and no longer have tensor semantics).
    SmallVector<linalg::LinalgOp> targets;
    getOperation()->walk([&](linalg::LinalgOp op) {
      if (op->hasAttr(cnm::CnmDialect::TILE_SIZES_NAME)) {
        if (isIsolatedFromAbove(op->getRegion(0))) {
          targets.push_back(op);
        } else {
          op.emitWarning("Cannot be converted to CNM as the body is not "
                         "IsolatedFromAbove");
        }
      }
    });

    DistributionOptions options{
        bufferLevel, leafTileAttr,      allowFloatReassociation,
        perDimAttrs, workgroupDimOrder, workgroupDimOrderIndex};

    IRRewriter rewriter(&getContext());
    for (linalg::LinalgOp op : targets)
      if (failed(distribute(rewriter, op, options)))
        return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::cnm::createConvertLinalgToCnmPass() {
  return std::make_unique<ConvertLinalgToCnmPass>();
}

std::unique_ptr<Pass>
mlir::cnm::createConvertLinalgToCnmPass(ConvertLinalgToCnmPassOptions options) {
  return std::make_unique<ConvertLinalgToCnmPass>(options);
}
