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
// Everything follows from one block-size vector, so unlike
// `--convert-cinm-to-cnm` this pass takes no decisions of its own.
//
//===----------------------------------------------------------------------===//

#include "cinm-mlir/Conversion/CnmBufferLevel.h"
#include "cinm-mlir/Conversion/CnmPasses.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmBase.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmOps.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmTypes.h"
#include "cinm-mlir/Utils/CinmUtils.h"

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
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>

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
    if (attr.getName() != cnm::CnmDialect::TILE_SIZES_NAME)
      state.addAttribute(attr.getName(), attr.getValue());
  for (Region &region : op->getRegions()) {
    auto cloned = std::make_unique<Region>();
    IRMapping mapping;
    region.cloneInto(cloned.get(), mapping);
    state.addRegion(std::move(cloned));
  }
  return b.create(state);
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
    auto shape = cast<ShapedType>(operand.get().getType()).getShape();
    for (auto [pos, expr] : llvm::enumerate(map.getResults()))
      extents[cast<AffineDimExpr>(expr).getPosition()] = shape[pos];
  }
  for (auto [dim, extent] : llvm::enumerate(extents))
    if (ShapedType::isDynamic(extent))
      return op->emitOpError("iteration dimension ")
             << dim << " has a dynamic extent; the distribution is computed "
                       "from static sizes";
  return extents;
}

/// Tile counts `extent / block`, checking that the block sizes make sense for
/// this op.
static FailureOr<SmallVector<int64_t>> getTileCounts(linalg::LinalgOp op,
                                                     ArrayRef<int64_t> blocks,
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
/// The split dimension is inserted *before* the parallel dimensions, making it
/// the outermost one. Under §G3's linearization the original parallel
/// dimensions then vary fastest across leaves, so the leaves sharing one node
/// of the workgroup (tasklets within a DPU, on UPMEM) differ in their parallel
/// tile and share their reduction tile. That is what lets a broadcast operand
/// indexed only by reduction dimensions -- gemv's vector -- be stored once per
/// node instead of replicated per leaf. See §G3 for the evidence and for what
/// it costs.
///
/// `blocks` is updated to describe the rewritten op.
static FailureOr<linalg::LinalgOp>
splitDistributedReductions(RewriterBase &rewriter, linalg::LinalgOp op,
                           SmallVector<int64_t> &blocks,
                           bool allowFloatReassociation,
                           ArrayRef<std::string> perDimAttrs) {
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
    Type elementType =
        cast<ShapedType>(op.getDpsInits()[0].getType()).getElementType();
    if (isa<FloatType>(elementType) && !allowFloatReassociation)
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
             perDimAttrs)) {
      auto attr = op->getAttrOfType<DenseI64ArrayAttr>(name);
      if (!attr || attr.size() != static_cast<int64_t>(oldNumLoops))
        continue;
      SmallVector<int64_t> updated(attr.asArrayRef());
      updated.insert(updated.begin(), 1);
      op->setAttr(name, DenseI64ArrayAttr::get(op->getContext(), updated));
    }

    FailureOr<SmallVector<int64_t>> newExtents = getLoopExtents(op);
    if (failed(newExtents))
      return failure();
    if ((*newExtents)[0] != ratio)
      return op->emitOpError("internal error: splitReduction placed the split "
                             "dimension somewhere unexpected");
  }
}

LogicalResult distribute(RewriterBase &rewriter, linalg::LinalgOp op,
                         StringRef bufferLevelName,
                         bool allowFloatReassociation,
                         ArrayRef<std::string> perDimAttrs) {
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
      cnm::resolveBufferLevel(bufferLevelName, accelerator, op);
  if (failed(level))
    return failure();

  if (failed(checkProjectedPermutations(op)))
    return failure();

  SmallVector<int64_t> blocks(tileAttr.asArrayRef());
  {
    FailureOr<SmallVector<int64_t>> extents = getLoopExtents(op);
    if (failed(extents))
      return failure();
    FailureOr<SmallVector<int64_t>> counts = getTileCounts(op, blocks, *extents);
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
  FailureOr<linalg::LinalgOp> split = splitDistributedReductions(
      rewriter, op, blocks, allowFloatReassociation, perDimAttrs);
  if (failed(split))
    return failure();
  op = *split;

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

  // Tile-space -> workgroup mapping. Both sides are linearized, and the
  // tile-side order is fixed by rule rather than searched: parallel
  // dimensions outer, reduction dimensions inner, op order within each group
  // (design §G3). Reduction dimensions innermost means the partials of one
  // output region occupy consecutive leaves, which is what makes them
  // mergeable locally once device-side merging exists (§G10).
  auto iteratorTypes = op.getIteratorTypesArray();
  SmallVector<unsigned> order;
  for (auto [dim, kind] : llvm::enumerate(iteratorTypes))
    if (kind == utils::IteratorType::parallel)
      order.push_back(dim);
  for (auto [dim, kind] : llvm::enumerate(iteratorTypes))
    if (kind != utils::IteratorType::parallel)
      order.push_back(dim);

  SmallVector<int64_t> strides(numLoops);
  int64_t stride = 1;
  for (unsigned dim : llvm::reverse(order)) {
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
  // a host element for every element of every leaf's buffer: the operand is
  // scattered as it stands, in whatever layout it already has.
  SmallVector<OperandTiling> tilings;
  for (auto [operand, map] : llvm::zip(op->getOpOperands(), indexingMaps)) {
    OperandTiling tiling;
    SmallVector<AffineExpr> scatterResults;
    unsigned operandRank = map.getNumResults();
    for (auto [position, expr] : llvm::enumerate(map.getResults())) {
      unsigned dim = cast<AffineDimExpr>(expr).getPosition();
      tiling.blocks.push_back(blocks[dim]);
      AffineExpr within =
          getAffineDimExpr(wgShape.size() + position, ctx);
      scatterResults.push_back(tileCoords[dim] * blocks[dim] + within);
    }
    SmallVector<int64_t> bounds(wgShape);
    llvm::append_range(bounds, tiling.blocks);
    tiling.scatterMap = simplifyAffineMapWithBounds(
        AffineMap::get(wgShape.size() + operandRank, 0, scatterResults, ctx),
        bounds);
    tilings.push_back(std::move(tiling));
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
  for (auto [operand, tiling] : llvm::zip(op->getOpOperands(), tilings)) {
    auto operandTy = cast<ShapedType>(operand.get().getType());
    auto bufferTy = cnm::BufferType::get(tiling.blocks,
                                         operandTy.getElementType(),
                                         accelerator, level->space);
    Value alloc = cnm::AllocOp::create(b, bufferTy, workgroup);
    bool isDestination = operand.getOperandNumber() >= numInputs;

    if (isDestination &&
        isa_and_nonnull<tensor::EmptyOp>(operand.get().getDefiningOp())) {
      // A freshly allocated destination has undefined contents, so there is
      // nothing to bring over.
    } else {
      // This scatter can then be optimized into a broadcast.
      // For now this happens in the backend dialect, which has knowledge of
      // the constraints on the scatter calls.
      cnm::ScatterOp::create(b, operand.get(), alloc, workgroup,
                             tiling.scatterMap);
    }

    (isDestination ? launchOutputs : launchInputs).push_back(alloc);
  }

  auto launch = cnm::LaunchOp::create(b, workgroup, launchInputs,
                                      launchOutputs);
  {
    Block &body = launch.getBody().emplaceBlock();
    for (Value param : launch.getParams()) {
      auto bufferTy = cast<cnm::BufferType>(param.getType());
      // The buffer's level becomes the memref's memory space; that is what
      // LaunchOp::verify requires, and it is how the body learns which
      // memory it computes on.
      body.addArgument(MemRefType::get(bufferTy.getShape(),
                                       bufferTy.getElementType(),
                                       MemRefLayoutAttrInterface{},
                                       bufferTy.getLevel()),
                       param.getLoc());
    }
    OpBuilder::InsertionGuard bodyGuard(b);
    b.setInsertionPointToStart(&body);
    cloneOnBuffers(b, op, body.getArguments().take_front(numInputs),
                   body.getArguments().drop_front(numInputs));
    cnm::TerminatorOp::create(b);
  }

  // Gather each result straight back into a value of the op's own shape.
  SmallVector<Value> results;
  for (auto [index, init] : llvm::enumerate(op.getDpsInits())) {
    const OperandTiling &tiling = tilings[numInputs + index];
    auto initTy = cast<ShapedType>(init.getType());
    Value destination = tensor::EmptyOp::create(b, initTy.getShape(),
                                                initTy.getElementType());
    results.push_back(cnm::GatherOp::create(b, launchOutputs[index], workgroup,
                                            tiling.scatterMap, destination)
                          .getOutput());
  }

  cnm::FreeWorkgroupOp::create(b, workgroup);
  rewriter.replaceOp(op, results);
  return success();
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
      if (op->hasAttr(cnm::CnmDialect::TILE_SIZES_NAME))
        targets.push_back(op);
    });

    IRRewriter rewriter(&getContext());
    for (linalg::LinalgOp op : targets)
      if (failed(distribute(rewriter, op, bufferLevel,
                            allowFloatReassociation, perDimAttrs)))
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
