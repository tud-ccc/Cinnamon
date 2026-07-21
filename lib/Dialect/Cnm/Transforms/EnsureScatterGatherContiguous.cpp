//===- EnsureScatterGatherContiguous.cpp - Pack non-contiguous transfers -===//
//
// Lowering cnm.scatter/cnm.gather to upmem.scatter/upmem.gather performs a
// single flat memcpy per DPU. This is only correct if the transferred
// elements are actually contiguous in the host memref (see
// upmem::ScatterOp::verify / upmem::GatherOp::verify and the runtime's
// do_dpu_transfer). This pass detects host memrefs that aren't contiguous
// (e.g. subviews of a larger tensor) and inserts an intermediate contiguous
// buffer, similar to what packATile does for the tiled GEMV/reduction
// templates, but generically for any cnm.scatter/cnm.gather.
//
//===----------------------------------------------------------------------===//

#include <cinm-mlir/Dialect/Cnm/IR/CnmOps.h>
#include <cinm-mlir/Dialect/Cnm/Transforms/Passes.h>
#include <cinm-mlir/Utils/CinmUtils.h>

#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>

namespace mlir::cnm {

#define GEN_PASS_DEF_CNMENSURESCATTERGATHERCONTIGUOUSPASS
#include <cinm-mlir/Dialect/Cnm/Transforms/Passes.h.inc>

} // namespace mlir::cnm

using namespace mlir;

namespace {

// Allocate a memref with the same shape/element type as `src` but with a
// default (fully packed, row-major) layout, resolving any dynamic dimensions
// from `src` itself.
Value allocateContiguousLike(OpBuilder &b, Location loc, Value src) {
  auto ty = cast<MemRefType>(src.getType());
  SmallVector<Value> dynSizes;
  for (int64_t i = 0; i < ty.getRank(); ++i)
    if (ty.isDynamicDim(i))
      dynSizes.push_back(memref::DimOp::create(b, loc, src, i));

  auto contiguousTy = MemRefType::get(ty.getShape(), ty.getElementType());
  return memref::AllocOp::create(b, loc, contiguousTy, dynSizes);
}

// `value` is contiguous enough for a flat per-DPU memcpy iff its whole shape
// (not just the per-workgroup-element buffer suffix) is packed row-major.
bool isFullyContiguous(TypedValue<ShapedType> value) {
  auto memrefTy = dyn_cast<MemRefType>(value.getType());
  if (!memrefTy)
    return true; // not yet bufferized; nothing we can (or need to) do here.
  return mlir::scatteredMemrefIsContiguous(value, memrefTy.getShape());
}

void ensureScatterContiguous(cnm::ScatterOp op, OpBuilder &b) {
  auto input = op.getInput();
  if (!isa<MemRefType>(input.getType()) || isFullyContiguous(input))
    return;

  Location loc = op.getLoc();
  b.setInsertionPoint(op);
  Value packed = allocateContiguousLike(b, loc, input);
  memref::CopyOp::create(b, loc, input, packed);
  op.getInputMutable().assign(packed);

  // b.setInsertionPointAfter(op);
  // memref::DeallocOp::create(b, loc, packed);
}

void ensureGatherContiguous(cnm::GatherOp op, OpBuilder &b) {
  auto outputBuf = op.getOutputBuf();
  if (!isa<MemRefType>(outputBuf.getType()) || isFullyContiguous(outputBuf))
    return;

  Location loc = op.getLoc();
  b.setInsertionPoint(op);
  Value packed = allocateContiguousLike(b, loc, outputBuf);
  op.getOutputBufMutable().assign(packed);

  b.setInsertionPointAfter(op);
  memref::CopyOp::create(b, loc, packed, outputBuf);
  memref::DeallocOp::create(b, loc, packed);
}

} // namespace

struct CnmEnsureScatterGatherContiguousPass
    : public cnm::impl::CnmEnsureScatterGatherContiguousPassBase<
          CnmEnsureScatterGatherContiguousPass> {
  void runOnOperation() override {
    OpBuilder builder(&getContext());

    getOperation()->walk([&](Operation *op) {
      if (auto scatter = dyn_cast<cnm::ScatterOp>(op))
        ensureScatterContiguous(scatter, builder);
      else if (auto gather = dyn_cast<cnm::GatherOp>(op))
        ensureGatherContiguous(gather, builder);
    });
  }
};
