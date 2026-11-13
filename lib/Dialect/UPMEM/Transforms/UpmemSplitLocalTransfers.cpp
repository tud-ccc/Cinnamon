//===- UpmemSplitLocalTransfers.cpp - One DMA per MRAM transfer -----------===//
//
// Splits an MRAM transfer longer than max-dma-bytes into consecutive
// transfers of at most that many bytes, each of which is one DMA.
//
//===----------------------------------------------------------------------===//

#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMDialect.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h"
#include "cinm-mlir/Dialect/UPMEM/Transforms/Passes.h"
#include "cinm-mlir/Utils/CinmUtils.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Arith/Utils/Utils.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Builders.h>

#include <numeric>

namespace mlir::upmem {

#define GEN_PASS_DEF_UPMEMSPLITLOCALTRANSFERSPASS
#include <cinm-mlir/Dialect/UPMEM/Transforms/Passes.h.inc>

namespace {

/// The DMA engine's granule and its longest transfer, in bytes.
constexpr int64_t kGranuleBytes = 8;
constexpr int64_t kMaxDmaBytes = 2048;

bool inSpace(Type type, DpuMemSpace space) {
  auto attr = dyn_cast_or_null<DpuMemSpaceAttr>(
      cast<MemRefType>(type).getMemorySpace());
  return attr && attr.getValue() == space;
}

/// One end of a transfer as a run of elements: the buffer it lies in, and
/// the element offset of its first element in it as (offset, stride) terms.
struct Run {
  Value buffer;
  SmallVector<std::pair<OpFoldResult, int64_t>> terms;
};

/// `view` as a run of its buffer: the buffer itself, or one subview of it
/// whose elements are contiguous. The buffer must have the identity layout,
/// so that it collapses to one dimension.
std::optional<Run> runOf(Value view) {
  auto type = cast<MemRefType>(view.getType());
  if (!type.hasStaticShape() || !memrefIsContiguous(type))
    return std::nullopt;
  auto identity = [](Value v) {
    return cast<MemRefType>(v.getType()).getLayout().isIdentity();
  };
  if (isa_and_nonnull<memref::AllocaOp, StaticAllocOp>(view.getDefiningOp()))
    return identity(view) ? std::optional<Run>(Run{view, {}}) : std::nullopt;
  auto subview = view.getDefiningOp<memref::SubViewOp>();
  if (!subview || !isa_and_nonnull<memref::AllocaOp, StaticAllocOp>(
                      subview.getSource().getDefiningOp()))
    return std::nullopt;
  if (!identity(subview.getSource()) ||
      !llvm::all_of(subview.getMixedStrides(),
                    [](OpFoldResult s) { return isConstantIntValue(s, 1); }))
    return std::nullopt;
  Run run{subview.getSource(), {}};
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(subview.getSourceType().getStridesAndOffset(strides, offset)))
    return std::nullopt;
  for (auto [off, stride] : llvm::zip_equal(subview.getMixedOffsets(), strides))
    if (!isConstantIntValue(off, 0))
      run.terms.push_back({off, stride});
  return run;
}

struct UpmemSplitLocalTransfersPass
    : impl::UpmemSplitLocalTransfersPassBase<UpmemSplitLocalTransfersPass> {
  using Base::Base;

  void runOnOperation() final {
    if (maxDmaBytes < kGranuleBytes || maxDmaBytes > kMaxDmaBytes ||
        maxDmaBytes % kGranuleBytes != 0) {
      getOperation()->emitError("max-dma-bytes must be a multiple of ")
          << kGranuleBytes << " between " << kGranuleBytes << " and "
          << kMaxDmaBytes << ", not " << maxDmaBytes.getValue();
      return signalPassFailure();
    }
    SmallVector<LocalTransferOp> transfers;
    getOperation()->walk([&](LocalTransferOp op) { transfers.push_back(op); });
    for (LocalTransferOp op : transfers)
      if (failed(split(op)))
        return signalPassFailure();
  }

  LogicalResult split(LocalTransferOp op) {
    Value src = op.getSource(), dst = op.getTarget();
    bool isDma = (inSpace(src.getType(), DpuMemSpace::MRAM) &&
                  inSpace(dst.getType(), DpuMemSpace::WRAM)) ||
                 (inSpace(src.getType(), DpuMemSpace::WRAM) &&
                  inSpace(dst.getType(), DpuMemSpace::MRAM));
    if (!isDma)
      return success();
    auto srcTy = cast<MemRefType>(src.getType());
    const int64_t elementBytes = srcTy.getElementTypeBitWidth() / 8;
    if (!srcTy.hasStaticShape() || elementBytes == 0 ||
        kGranuleBytes % elementBytes != 0)
      return success();
    const int64_t elements = srcTy.getNumElements();
    const int64_t bytes = elements * elementBytes;
    if (bytes <= maxDmaBytes)
      return success();

    std::optional<Run> srcRun = runOf(src), dstRun = runOf(dst);
    if (!srcRun || !dstRun || bytes % kGranuleBytes != 0) {
      if (bytes <= kMaxDmaBytes)
        return success();
      return op.emitOpError("moves ")
             << bytes << " bytes, more than one DMA's " << kMaxDmaBytes
             << ", and cannot be split: each end has to be a buffer or one "
                "contiguous subview of one, and the length whole "
             << kGranuleBytes << "-byte granules";
    }

    OpBuilder b(op);
    Location loc = op.getLoc();
    auto index = [&](int64_t v) -> Value {
      return arith::ConstantIndexOp::create(b, loc, v);
    };
    // The buffer as one dimension, and the run's first element in it.
    auto flatten = [&](const Run &run) -> std::pair<Value, Value> {
      Value flat = run.buffer;
      auto type = cast<MemRefType>(flat.getType());
      if (type.getRank() != 1) {
        ReassociationIndices all(type.getRank());
        std::iota(all.begin(), all.end(), 0);
        flat = memref::CollapseShapeOp::create(
            b, loc, flat, ArrayRef<ReassociationIndices>{all});
      }
      Value start = index(0);
      for (auto [off, stride] : run.terms)
        start = b.createOrFold<arith::AddIOp>(
            loc, start,
            b.createOrFold<arith::MulIOp>(
                loc, getValueOrCreateConstantIndexOp(b, loc, off),
                index(stride)));
      return {flat, start};
    };
    auto [srcFlat, srcStart] = flatten(*srcRun);
    auto [dstFlat, dstStart] = flatten(*dstRun);

    auto chunk = [&](Value flat, Value start, int64_t at, int64_t size) {
      SmallVector<OpFoldResult> offsets{
          at ? OpFoldResult(
                   b.createOrFold<arith::AddIOp>(loc, start, index(at)))
             : OpFoldResult(start)},
          sizes{b.getIndexAttr(size)}, strides{b.getIndexAttr(1)};
      auto type = memref::SubViewOp::inferResultType(
          cast<MemRefType>(flat.getType()), offsets, sizes, strides);
      return memref::SubViewOp::create(b, loc, cast<MemRefType>(type), flat,
                                       offsets, sizes, strides);
    };
    const int64_t perChunk = maxDmaBytes / elementBytes;
    for (int64_t at = 0; at < elements; at += perChunk) {
      int64_t size = std::min(perChunk, elements - at);
      LocalTransferOp::create(b, loc, chunk(srcFlat, srcStart, at, size),
                              chunk(dstFlat, dstStart, at, size));
    }

    op.erase();
    for (Value end : {src, dst})
      if (auto view = end.getDefiningOp<memref::SubViewOp>())
        if (view->use_empty())
          view.erase();
    return success();
  }
};

} // namespace
} // namespace mlir::upmem
