//===- UpmemTileMRAMBuffers.cpp - Stage launch bodies down to the leaf level =//
//
// Tiles the ops in a cnm.launch body down to a leaf-level tile and stages the
// tile through the leaf memory level, wrapping it in cnm.local_transfers.
//
//===----------------------------------------------------------------------===//

#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmDialect.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmInterfaces.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmOps.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMDialect.h"
#include "cinm-mlir/Dialect/UPMEM/Transforms/Passes.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/Transforms/TileUsingInterface.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/TilingInterface.h>

namespace mlir::upmem {

#define GEN_PASS_DEF_UPMEMTILEMRAMBUFFERSPASS
#include <cinm-mlir/Dialect/UPMEM/Transforms/Passes.h.inc>

namespace {

/// The memory levels of the workgroup a launch body runs on, as seen from a
/// compute element.
struct LevelHierarchy {
  /// The level closest to the compute elements -- the only one they can
  /// address directly, and therefore the one everything must be staged into.
  cinm::CinmLevelDefAttr leaf;
  /// Memory space attribute denoting `leaf`.
  cinm::CinmLevelAttrInterface leafSpace;
  cnm::CnmAcceleratorAttrInterface accelerator;

  /// Whether `memspace` denotes a level that is not the leaf, i.e. a buffer
  /// there has to be staged before the compute elements can work on it.
  bool isNonLeaf(Attribute memspace) const {
    cinm::CinmLevelDefAttr level =
        accelerator.getPlatform().getLevelOfMemspace(memspace);
    return level && level != leaf;
  }
};

/// Read the level hierarchy off the launch a body op belongs to.
///
/// `getWorkgroupMemoryLevels()` is indexed by workgroup dimension, each entry
/// listing that dimension's levels ordered from farthest to closest to the
/// compute elements. The leaf level is therefore the last entry of the
/// innermost dimension that declares any.
std::optional<LevelHierarchy> getLevelHierarchy(Operation *op) {
  auto launch = op->getParentOfType<cnm::LaunchOp>();
  if (!launch)
    return std::nullopt;

  auto accelerator = launch.getWg().getType().getAccelerator();
  auto platform = accelerator.getPlatform();
  if (!platform)
    return std::nullopt;

  cinm::CinmLevelDefAttr leaf;
  for (cinm::CinmLevelArrayAttr levels :
       accelerator.getWorkgroupMemoryLevels())
    if (!levels.empty())
      leaf = levels.back();
  if (!leaf)
    return std::nullopt;

  cinm::CinmLevelAttrInterface leafSpace = platform.getMemrefMemspace(leaf);
  if (!leafSpace)
    return std::nullopt;

  return LevelHierarchy{leaf, leafSpace, accelerator};
}

/// Operand indices whose buffers are in a non-leaf level, i.e. the ones that
/// have to be staged. Empty means there is nothing for this pass to do.
SmallVector<int64_t> operandsToStage(linalg::LinalgOp op,
                                     const LevelHierarchy &levels) {
  SmallVector<int64_t> result;
  for (OpOperand &operand : op->getOpOperands()) {
    auto memrefTy = dyn_cast<MemRefType>(operand.get().getType());
    if (memrefTy && levels.isNonLeaf(memrefTy.getMemorySpace()))
      result.push_back(operand.getOperandNumber());
  }
  return result;
}

/// Tile sizes for `op`: its own `upmem.leaf_tile_sizes` attribute if it has
/// one (that is how a resolved configuration is passed in), else the pass
/// option. Empty means "do not tile", which is right when the buffer already
/// fits the leaf level.
SmallVector<int64_t> getTileSizes(linalg::LinalgOp op,
                                  ArrayRef<int64_t> fallback) {
  if (auto attr = op->getAttrOfType<DenseI64ArrayAttr>(
          UPMEMDialect::LEAF_TILE_SIZES_NAME))
    return SmallVector<int64_t>(attr.asArrayRef());
  return SmallVector<int64_t>(fallback);
}

/// Allocate a promoted buffer in the leaf level.
///
/// Supplying this is not optional: the default scheme allocates a flat
/// `memref<Nxi8>` and takes a `memref.view` of it, which yields dynamically
/// shaped buffers. The UPMEM backend needs static shapes to turn these into
/// its own allocations, and a dynamically shaped WRAM buffer could not be
/// sized against the level's capacity either.
std::optional<Value> allocateInLeaf(OpBuilder &b, memref::SubViewOp subView,
                                    ArrayRef<Value> boundingSubViewSize,
                                    const LevelHierarchy &levels) {
  SmallVector<int64_t> shape;
  for (Value size : boundingSubViewSize) {
    APInt constant;
    if (!matchPattern(size, m_ConstantInt(&constant)))
      return std::nullopt;
    shape.push_back(constant.getSExtValue());
  }

  auto type = MemRefType::get(shape, subView.getType().getElementType(),
                              MemRefLayoutAttrInterface{}, levels.leafSpace);
  return memref::AllocOp::create(b, subView.getLoc(), type).getResult();
}

/// promoteSubViews only promotes operands defined by a `memref.subview`, which
/// is what tiling produces. When the op was not tiled -- because the buffer
/// already fits the leaf level -- wrap each operand in a full-extent subview so
/// the same path applies.
LogicalResult materializeIdentitySubviews(IRRewriter &rewriter,
                                          linalg::LinalgOp op,
                                          ArrayRef<int64_t> toStage) {
  rewriter.setInsertionPoint(op);
  for (int64_t idx : toStage) {
    OpOperand &operand = op->getOpOperand(idx);
    if (operand.get().getDefiningOp<memref::SubViewOp>())
      continue;

    auto type = cast<MemRefType>(operand.get().getType());
    if (!type.hasStaticShape())
      return op->emitOpError("cannot stage operand ")
             << idx << " of type " << type
             << " into the leaf memory level: its shape is not static, so the "
                "size of the staging buffer is unknown";

    SmallVector<OpFoldResult> offsets(type.getRank(), rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(type.getRank(), rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> sizes;
    for (int64_t dim : type.getShape())
      sizes.push_back(rewriter.getIndexAttr(dim));

    Value subview = memref::SubViewOp::create(rewriter, op.getLoc(),
                                              operand.get(), offsets, sizes,
                                              strides);
    rewriter.modifyOpInPlace(op, [&] { operand.set(subview); });
  }
  return success();
}

/// Whether `subView` covers the whole of its source with unit strides, so that
/// using the source directly denotes the same memory.
bool isIdentitySubview(memref::SubViewOp subView) {
  auto sourceType = subView.getSourceType();
  if (!sourceType.hasStaticShape() ||
      subView.getType().getRank() != sourceType.getRank())
    return false;

  auto isConstant = [](OpFoldResult ofr, int64_t value) {
    std::optional<int64_t> constant = getConstantIntValue(ofr);
    return constant && *constant == value;
  };
  if (!llvm::all_of(subView.getMixedOffsets(),
                    [&](OpFoldResult o) { return isConstant(o, 0); }) ||
      !llvm::all_of(subView.getMixedStrides(),
                    [&](OpFoldResult s) { return isConstant(s, 1); }))
    return false;
  return llvm::all_of(
      llvm::zip_equal(subView.getMixedSizes(), sourceType.getShape()),
      [&](auto pair) {
        return isConstant(std::get<0>(pair), std::get<1>(pair));
      });
}

/// Promotion hands the op a *partial* view of the staging buffer, which for a
/// full tile covers the buffer entirely but is typed dynamically. Some linalg
/// ops fold that away later (contract and matmul have memref-cast folders),
/// but others do not -- `linalg.reduce`, for one -- and a dynamically shaped
/// launch body is one the backend cannot size a transfer or an allocation
/// from. Drop the identity view uniformly instead of relying on the op.
void useStagingBuffersDirectly(linalg::LinalgOp op) {
  for (OpOperand &operand : op->getOpOperands()) {
    auto subView = operand.get().getDefiningOp<memref::SubViewOp>();
    if (subView && isIdentitySubview(subView))
      operand.set(subView.getSource());
  }
}

struct UpmemTileMRAMBuffersPass
    : public impl::UpmemTileMRAMBuffersPassBase<UpmemTileMRAMBuffersPass> {
  using Base::Base;

  void runOnOperation() override {
    // Collect first: tiling and promotion both rewrite the ops in place.
    SmallVector<linalg::LinalgOp> candidates;
    getOperation()->walk([&](linalg::LinalgOp op) {
      if (op.hasPureBufferSemantics())
        candidates.push_back(op);
    });

    IRRewriter rewriter(&getContext());
    for (linalg::LinalgOp op : candidates)
      if (failed(stageOp(rewriter, op)))
        return signalPassFailure();
  }

  LogicalResult stageOp(IRRewriter &rewriter, linalg::LinalgOp op) {
    std::optional<LevelHierarchy> levels = getLevelHierarchy(op);
    if (!levels)
      return success(); // not in a launch we know the memory hierarchy of
    if (operandsToStage(op, *levels).empty())
      return success(); // already entirely in the leaf level

    SmallVector<int64_t> sizes = getTileSizes(op, tileSizes);
    if (!sizes.empty()) {
      auto tileable = cast<TilingInterface>(op.getOperation());
      if (sizes.size() != tileable.getLoopIteratorTypes().size())
        return op->emitOpError("expected ")
               << tileable.getLoopIteratorTypes().size()
               << " tile sizes for a " << op->getName() << " with "
               << tileable.getLoopIteratorTypes().size()
               << " iteration dimensions, got " << sizes.size();

      scf::SCFTilingOptions options;
      options.setTileSizes(getAsIndexOpFoldResult(&getContext(), sizes));
      rewriter.setInsertionPoint(op);
      FailureOr<scf::SCFTilingResult> tiled =
          scf::tileUsingSCF(rewriter, tileable, options);
      if (failed(tiled))
        return op->emitOpError("failed to tile for the leaf memory level");
      rewriter.eraseOp(op);
      op = cast<linalg::LinalgOp>(tiled->tiledOps.back());
      // The op's operands are now subviews, which is what promotion needs.
      op->removeAttr(UPMEMDialect::LEAF_TILE_SIZES_NAME); // consumed
    }

    return promote(rewriter, op, *levels);
  }

  LogicalResult promote(IRRewriter &rewriter, linalg::LinalgOp op,
                        const LevelHierarchy &levels) {
    SmallVector<int64_t> toStage = operandsToStage(op, levels);
    if (toStage.empty())
      return success();
    if (failed(materializeIdentitySubviews(rewriter, op, toStage)))
      return failure();

    linalg::LinalgPromotionOptions options;
    options.setOperandsToPromote(toStage)
        .setMemorySpace(levels.leafSpace)
        .setUseFullTileBuffersByDefault(false)
        .setAllocationDeallocationFns(
            [&levels](OpBuilder &b, memref::SubViewOp subView,
                      ArrayRef<Value> sizes,
                      DataLayout &) -> std::optional<Value> {
              return allocateInLeaf(b, subView, sizes, levels);
            },
            [](OpBuilder &b, Value buffer) -> LogicalResult {
              memref::DeallocOp::create(b, buffer.getLoc(), buffer);
              return success();
            })
        .setCopyInOutFns(
            [](OpBuilder &b, Value src, Value dst) -> LogicalResult {
              cnm::LocalTransferOp::create(b, src.getLoc(), src, dst);
              return success();
            },
            [](OpBuilder &b, Value src, Value dst) -> LogicalResult {
              cnm::LocalTransferOp::create(b, src.getLoc(), src, dst);
              return success();
            });

    if (failed(linalg::promoteSubviewsPrecondition(op, options)))
      return op->emitOpError(
          "operands are in a non-leaf memory level but cannot be staged; "
          "expected each of them to be a memref.subview, which tiling "
          "produces");

    rewriter.setInsertionPoint(op);
    FailureOr<linalg::LinalgOp> promoted =
        linalg::promoteSubViews(rewriter, op, options);
    if (failed(promoted))
      return op->emitOpError("failed to stage operands into the leaf level");
    useStagingBuffersDirectly(*promoted);
    return success();
  }
};

} // namespace
} // namespace mlir::upmem
