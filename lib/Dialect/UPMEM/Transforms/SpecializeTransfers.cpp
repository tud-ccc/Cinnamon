//===- SpecializeTransfers.cpp - Narrow transfer ops to cheaper forms ----===//
//
// Emitters produce the general block form and this pass narrows it. See the
// pass description in Passes.td for the three rewrites and why they live
// here rather than in each emitter.
//
//===----------------------------------------------------------------------===//

#include <cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h>
#include <cinm-mlir/Dialect/UPMEM/Transforms/Passes.h>
#include <cinm-mlir/Utils/CinmUtils.h>

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace mlir::upmem {

#define GEN_PASS_DEF_UPMEMSPECIALIZETRANSFERSPASS
#include <cinm-mlir/Dialect/UPMEM/Transforms/Passes.h.inc>

} // namespace mlir::upmem

using namespace mlir;

namespace {

/// The `upmem.static_alloc` a transfer op names, or null when the hierarchy
/// cannot be resolved statically.
template <class Op> upmem::StaticAllocOp getTargetBuffer(Op op) {
  upmem::DpuProgramOp program = op.getDpuProgram();
  if (!program)
    return {};
  return dyn_cast_or_null<upmem::StaticAllocOp>(
      SymbolTable::lookupSymbolIn(program, op.getDpuBufRefAttr().getAttr()));
}

/// Carry over the labels a transfer may have picked up (upmem.timing_tag and
/// the like) without touching the new op's own operands and attributes.
void inheritLabels(Operation *replacement, Operation *original) {
  replacement->setDiscardableAttrs(original->getDiscardableAttrDictionary());
}

/// A `memref.get_global` of a private constant of shape `shape` all of whose
/// elements are `uniform`, created once per (shape, value) in `module`.
Value materializeUniformConstant(RewriterBase &rewriter, Location loc,
                                 ModuleOp module, ArrayRef<int64_t> shape,
                                 TypedAttr uniform) {
  auto tileTy = MemRefType::get(shape, uniform.getType());
  // A splat DenseElementsAttr stores one element whatever the shape, so the
  // widening this pass does costs nothing until the backend materializes it.
  auto contents = DenseElementsAttr::get(
      RankedTensorType::get(shape, uniform.getType()), uniform);

  memref::GlobalOp global;
  for (auto candidate : module.getOps<memref::GlobalOp>()) {
    if (candidate.getConstant() && candidate.getType() == tileTy &&
        candidate.getInitialValueAttr() == contents) {
      global = candidate;
      break;
    }
  }

  if (!global) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    global = memref::GlobalOp::create(
        rewriter, loc, "__upmem_broadcast_tile",
        /*sym_visibility=*/rewriter.getStringAttr("private"), tileTy, contents,
        /*constant=*/true, /*alignment=*/IntegerAttr{});
    SymbolTable(module).insert(global);
  }
  return memref::GetGlobalOp::create(rewriter, loc, tileTy,
                                     global.getSymNameAttr());
}

/// Scattering a value whose every element is the same constant puts that
/// constant everywhere the map reaches, so where each block came from stops
/// mattering: one broadcast of a constant the size of the target buffer
/// delivers the same bytes. The blocks a DPU receives are laid out back to
/// back in MRAM, which is what makes the widened constant the right shape.
template <class Op>
Operation *broadcastUniformValue(Op op, RewriterBase &rewriter) {
  std::optional<TypedAttr> uniform = getUniformValue(op.getHostBuffer());
  if (!uniform)
    return nullptr;

  upmem::StaticAllocOp target = getTargetBuffer(op);
  if (!target)
    return nullptr;
  MemRefType targetTy = target.getType();
  if (!targetTy.hasStaticShape() ||
      targetTy.getElementType() != uniform->getType())
    return nullptr;
  // A broadcast fills the buffer from its start, so the transfer has to be
  // the whole of it -- otherwise it would write bytes the scatter did not.
  int64_t transferred = static_cast<int64_t>(op.getNumBlocksPerDpu()) *
                        static_cast<int64_t>(op.getTransferCount());
  if (targetTy.getNumElements() != transferred)
    return nullptr;

  rewriter.setInsertionPoint(op);
  Value tile = materializeUniformConstant(
      rewriter, op.getLoc(), op->template getParentOfType<ModuleOp>(),
      targetTy.getShape(), *uniform);
  auto broadcast = upmem::BroadcastOp::create(
      rewriter, op.getLoc(), tile, op.getDpuBufRefAttr(), op.getHierarchy());
  inheritLabels(broadcast, op);
  rewriter.eraseOp(op);
  return broadcast;
}

/// A map that ignores rank and DPU sends every DPU the same region. When that
/// region is the whole host buffer, starting at its first element, a
/// broadcast delivers it without widening anything.
Operation *broadcastWholeBuffer(upmem::ScatterOnArrayOp op,
                                RewriterBase &rewriter) {
  MemRefType hostTy = op.getHostBuffer().getType();
  if (!hostTy.hasStaticShape() ||
      hostTy.getNumElements() != static_cast<int64_t>(op.getTransferCount()))
    return nullptr;
  AffineMap map = op.getScatterMap();
  if (!map.isConstant() || !llvm::all_of(map.getConstantResults(),
                                         [](int64_t v) { return v == 0; }))
    return nullptr;

  rewriter.setInsertionPoint(op);
  auto broadcast =
      upmem::BroadcastOp::create(rewriter, op.getLoc(), op.getHostBuffer(),
                                 op.getDpuBufRefAttr(), op.getHierarchy());
  inheritLabels(broadcast, op);
  rewriter.eraseOp(op);
  return broadcast;
}

/// Whether every DPU's blocks form a single run: the element offset of block
/// `b` advances by exactly one block per step, so the `numBlocksPerDpu`
/// blocks are adjacent and in order.
bool blocksAreOneRun(AffineMap map, MemRefType hostTy, int64_t blockSize) {
  FailureOr<AffineExpr> offset = linearizeToElementOffset(map, hostTy);
  if (failed(offset))
    return false;

  MLIRContext *ctx = map.getContext();
  SmallVector<AffineExpr> atZero, atOne;
  for (unsigned i = 0; i < map.getNumDims(); ++i) {
    atZero.push_back(getAffineDimExpr(i, ctx));
    atOne.push_back(getAffineDimExpr(i, ctx));
  }
  atZero[2] = getAffineConstantExpr(0, ctx);
  atOne[2] = getAffineConstantExpr(1, ctx);

  AffineExpr step = simplifyAffineExpr(
      offset->replaceDims(atOne) - offset->replaceDims(atZero),
      map.getNumDims(), 0);
  auto stride = dyn_cast<AffineConstantExpr>(step);
  return stride && stride.getValue() == blockSize;
}

/// The (rank, dpu, block) map with the block dimension pinned to its first
/// value: where the run starts.
AffineMap dropBlockDim(AffineMap map) {
  MLIRContext *ctx = map.getContext();
  SmallVector<AffineExpr> substitution{getAffineDimExpr(0, ctx),
                                       getAffineDimExpr(1, ctx),
                                       getAffineConstantExpr(0, ctx)};
  SmallVector<AffineExpr> results;
  for (AffineExpr e : map.getResults())
    results.push_back(e.replaceDims(substitution));
  return AffineMap::get(2, map.getNumSymbols(), results, ctx);
}

/// Adjacent, in-order blocks are one contiguous run, which the flat per-DPU
/// transfer moves in one go -- no scatter/gather descriptors needed.
template <class FlatOp, class BlockOp>
Operation *collapseBlocksToOneRun(BlockOp op, RewriterBase &rewriter) {
  MemRefType hostTy = op.getHostBuffer().getType();
  int64_t blockSize = op.getTransferCount();
  int64_t total = static_cast<int64_t>(op.getNumBlocksPerDpu()) * blockSize;
  if (op.getNumBlocksPerDpu() > 1 &&
      !blocksAreOneRun(op.getScatterMap(), hostTy, blockSize))
    return nullptr;
  // The flat form's own contract: the whole run must be contiguous, which a
  // one-block-at-a-time transfer never had to be.
  int64_t contiguous = getContiguousSuffixSize(hostTy);
  if (contiguous >= 0 && total > contiguous)
    return nullptr;

  rewriter.setInsertionPoint(op);
  auto flat = FlatOp::create(rewriter, op.getLoc(), op.getHostBuffer(),
                             op.getDpuBufRefAttr(),
                             rewriter.getI64IntegerAttr(total),
                             AffineMapAttr::get(dropBlockDim(op.getScatterMap())),
                             op.getHierarchy());
  inheritLabels(flat, op);
  rewriter.eraseOp(op);
  return flat;
}

/// Broadcasting a widened uniform constant, for either form a scatter can
/// take. Tried before collapsing: a transfer that qualifies does not care
/// whether its blocks happened to be adjacent.
template <class Op> struct BroadcastUniformValue : OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    return success(broadcastUniformValue(op, rewriter) != nullptr);
  }
};

struct BroadcastWholeBuffer : OpRewritePattern<upmem::ScatterOnArrayOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(upmem::ScatterOnArrayOp op,
                                PatternRewriter &rewriter) const override {
    return success(broadcastWholeBuffer(op, rewriter) != nullptr);
  }
};

template <class FlatOp, class BlockOp>
struct CollapseBlocksToOneRun : OpRewritePattern<BlockOp> {
  using OpRewritePattern<BlockOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BlockOp op,
                                PatternRewriter &rewriter) const override {
    return success(collapseBlocksToOneRun<FlatOp>(op, rewriter) != nullptr);
  }
};

} // namespace

struct UpmemSpecializeTransfersPass
    : public upmem::impl::UpmemSpecializeTransfersPassBase<
          UpmemSpecializeTransfersPass> {
  using UpmemSpecializeTransfersPassBase::UpmemSpecializeTransfersPassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    // Gathers never narrow to a broadcast: two DPUs writing one host region
    // is a race, not a broadcast.
    patterns.add<CollapseBlocksToOneRun<upmem::ScatterOnArrayOp,
                                        upmem::ScatterBlocksOp>,
                 CollapseBlocksToOneRun<upmem::GatherFromArrayOp,
                                        upmem::GatherBlocksOp>>(&getContext());
    if (useBcXferCodegen)
      patterns.add<BroadcastUniformValue<upmem::ScatterBlocksOp>,
                   BroadcastUniformValue<upmem::ScatterOnArrayOp>,
                   BroadcastWholeBuffer>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();

    if (useSgXferCodegen)
      return;
    // The scatter/gather transfer API is off, so every block form was
    // supposed to have been made flat -- either here, or by packing the host
    // buffer upstream (--cnm-ensure-scatter-gather-contiguous). One that
    // survived would otherwise be silently collapsed into a transfer of the
    // wrong bytes.
    getOperation()->walk([&](Operation *op) {
      if (isa<upmem::ScatterBlocksOp, upmem::GatherBlocksOp>(op)) {
        op->emitOpError("cannot be narrowed to a single per-DPU block, but "
                        "the scatter/gather transfer API is disabled; its "
                        "host buffer needed packing upstream");
        signalPassFailure();
      }
    });
  }
};
