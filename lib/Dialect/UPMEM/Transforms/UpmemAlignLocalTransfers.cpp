//===- UpmemAlignLocalTransfers.cpp - Granule-sized MRAM transfers --------===//
//
// Rewrites an MRAM transfer that is shorter than a DMA granule, or does not
// start on one, into a transfer of the aligned granule that holds it, staged
// through a granule-sized WRAM buffer.
//
//===----------------------------------------------------------------------===//

#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMDialect.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h"
#include "cinm-mlir/Dialect/UPMEM/Transforms/Passes.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Arith/Utils/Utils.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Builders.h>

#include <numeric>

namespace mlir::upmem {

#define GEN_PASS_DEF_UPMEMALIGNLOCALTRANSFERSPASS
#include <cinm-mlir/Dialect/UPMEM/Transforms/Passes.h.inc>

namespace {

constexpr int64_t kGranuleBytes = 8;

bool inSpace(Type type, DpuMemSpace space) {
  auto attr = dyn_cast_or_null<DpuMemSpaceAttr>(
      cast<MemRefType>(type).getMemorySpace());
  return attr && attr.getValue() == space;
}

/// What an index is provably a multiple of: the reasoning of the C
/// translator's alignment check, which the rewritten transfers must pass.
int64_t knownMultipleOf(OpFoldResult ofr) {
  if (std::optional<int64_t> constant = getConstantIntValue(ofr))
    return *constant ? std::abs(*constant) : 0;
  auto value = dyn_cast<Value>(ofr);
  Operation *def = value ? value.getDefiningOp() : nullptr;
  if (!def)
    return 1;
  if (auto mul = dyn_cast<arith::MulIOp>(def))
    return knownMultipleOf(mul.getLhs()) * knownMultipleOf(mul.getRhs());
  if (isa<arith::AddIOp, arith::SubIOp>(def))
    return std::gcd(knownMultipleOf(def->getOperand(0)),
                    knownMultipleOf(def->getOperand(1)));
  return 1;
}

/// The MRAM side of a transfer as the translator addresses it: a static
/// buffer, and the element offset into it as (offset, stride) terms.
struct MramEndpoint {
  StaticAllocOp buffer;
  SmallVector<std::pair<OpFoldResult, int64_t>> terms;
  /// The byte offset is a multiple of this; 0 when it is zero.
  int64_t alignBytes = 0;
};

std::optional<MramEndpoint> mramEndpoint(Value view, int64_t elementBytes) {
  MramEndpoint e;
  if (auto alloc = view.getDefiningOp<StaticAllocOp>()) {
    e.buffer = alloc;
    return e;
  }
  auto subview = view.getDefiningOp<memref::SubViewOp>();
  if (!subview)
    return std::nullopt;
  e.buffer = subview.getSource().getDefiningOp<StaticAllocOp>();
  if (!e.buffer)
    return std::nullopt;
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(subview.getSourceType().getStridesAndOffset(strides, offset)) ||
      offset != 0)
    return std::nullopt;
  for (auto [off, stride] :
       llvm::zip_equal(subview.getMixedOffsets(), strides)) {
    if (isConstantIntValue(off, 0))
      continue;
    e.terms.push_back({off, stride});
    e.alignBytes =
        std::gcd(e.alignBytes, knownMultipleOf(off) * stride * elementBytes);
  }
  return e;
}

/// Whether the calling tasklet is the only writer of the granules of
/// `buffer` it touches: it is the program's one tasklet, or the buffer's
/// leading dimension is indexed by the tasklet and a slice is whole granules.
bool ownsGranules(StaticAllocOp buffer, Value view, int64_t elementBytes) {
  if (buffer.getNumSlots() > 1)
    return false;
  if (buffer->getParentOfType<DpuProgramOp>().getNumTasklets() == 1)
    return true;
  auto subview = view.getDefiningOp<memref::SubViewOp>();
  auto lead =
      subview ? dyn_cast<Value>(subview.getMixedOffsets().front()) : Value();
  if (!lead || !lead.getDefiningOp<TaskletDimOp>())
    return false;
  MemRefType type = buffer.getBuffer().getType();
  int64_t sliceBytes =
      type.getNumElements() / type.getDimSize(0) * elementBytes;
  return sliceBytes % kGranuleBytes == 0;
}

struct UpmemAlignLocalTransfersPass
    : impl::UpmemAlignLocalTransfersPassBase<UpmemAlignLocalTransfersPass> {
  void runOnOperation() final {
    SmallVector<LocalTransferOp> transfers;
    getOperation()->walk([&](LocalTransferOp op) { transfers.push_back(op); });
    for (LocalTransferOp op : transfers)
      rewrite(op);
  }

  void rewrite(LocalTransferOp op) {
    bool isRead = inSpace(op.getSource().getType(), DpuMemSpace::MRAM) &&
                  inSpace(op.getTarget().getType(), DpuMemSpace::WRAM);
    bool isWrite = inSpace(op.getSource().getType(), DpuMemSpace::WRAM) &&
                   inSpace(op.getTarget().getType(), DpuMemSpace::MRAM);
    if (!isRead && !isWrite)
      return;
    Value mram = isRead ? op.getSource() : op.getTarget();
    Value wram = isRead ? op.getTarget() : op.getSource();
    auto mramTy = cast<MemRefType>(mram.getType());
    const int64_t elementBytes = mramTy.getElementTypeBitWidth() / 8;
    if (!mramTy.hasStaticShape() || elementBytes == 0 ||
        kGranuleBytes % elementBytes != 0)
      return;
    const int64_t bytes = mramTy.getNumElements() * elementBytes;
    std::optional<MramEndpoint> e = mramEndpoint(mram, elementBytes);
    if (!e)
      return;
    const bool aligned = e->alignBytes % kGranuleBytes == 0;
    if (aligned && bytes % kGranuleBytes == 0)
      return;
    // A short read into a whole WRAM buffer is one the translator rounds up
    // into the buffer's own padding.
    if (isRead && aligned && !wram.getDefiningOp<memref::SubViewOp>())
      return;
    // Only data inside one granule: its cover is that granule, whatever the
    // offset at run time.
    if (bytes > std::gcd(e->alignBytes, kGranuleBytes))
      return;
    if (isWrite && !ownsGranules(e->buffer, mram, elementBytes))
      return;
    MemRefType bufTy = e->buffer.getBuffer().getType();
    const int64_t perGranule = kGranuleBytes / elementBytes;
    // The granule view stays inside the buffer's type.
    if (bufTy.getRank() == 0 || bufTy.getNumElements() % perGranule != 0)
      return;

    OpBuilder b(op);
    Location loc = op.getLoc();
    MLIRContext *ctx = op.getContext();
    Type elt = bufTy.getElementType();

    // The buffer as one run of elements, which is how the translator
    // addresses it anyway.
    Value flat = e->buffer.getBuffer();
    if (bufTy.getRank() > 1) {
      ReassociationIndices all(bufTy.getRank());
      std::iota(all.begin(), all.end(), 0);
      flat = memref::CollapseShapeOp::create(
          b, loc, flat, ArrayRef<ReassociationIndices>{all});
    }

    auto index = [&](int64_t v) -> Value {
      return arith::ConstantIndexOp::create(b, loc, v);
    };
    Value linear = index(0);
    for (auto [off, stride] : e->terms)
      linear = arith::AddIOp::create(
          b, loc, linear,
          arith::MulIOp::create(b, loc,
                                getValueOrCreateConstantIndexOp(b, loc, off),
                                index(stride)));
    Value granule = arith::MulIOp::create(
        b, loc, arith::DivUIOp::create(b, loc, linear, index(perGranule)),
        index(perGranule));
    Value within = arith::SubIOp::create(b, loc, linear, granule);

    auto viewOf = [&](Value base, Value offset, int64_t size) -> Value {
      SmallVector<OpFoldResult> offsets{offset}, sizes{b.getIndexAttr(size)},
          strides{b.getIndexAttr(1)};
      auto ty = memref::SubViewOp::inferResultType(
          cast<MemRefType>(base.getType()), offsets, sizes, strides);
      return memref::SubViewOp::create(b, loc, cast<MemRefType>(ty), base,
                                       offsets, sizes, strides);
    };
    Value mramGranule = viewOf(flat, granule, perGranule);
    auto bounceTy =
        MemRefType::get({perGranule}, elt, nullptr,
                        DpuMemSpaceAttr::get(ctx, DpuMemSpace::WRAM));
    Value bounce = memref::AllocaOp::create(b, loc, bounceTy);
    Value data = viewOf(bounce, within, bytes / elementBytes);

    LocalTransferOp::create(b, loc, mramGranule, bounce);
    if (isRead) {
      LocalTransferOp::create(b, loc, data, wram);
    } else {
      LocalTransferOp::create(b, loc, wram, data);
      LocalTransferOp::create(b, loc, bounce, mramGranule);
    }
    op.erase();
    if (auto view = mram.getDefiningOp<memref::SubViewOp>())
      if (view->use_empty())
        view.erase();
  }
};

} // namespace
} // namespace mlir::upmem
