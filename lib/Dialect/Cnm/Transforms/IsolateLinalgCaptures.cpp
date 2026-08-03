//===- IsolateLinalgCaptures.cpp - Turn captures into operands -----------===//
//
// Implements `--cnm-isolate-linalg-captures`; see the pass description in
// Passes.td for the motivating example (a fused `tensor.splat` factor) and
// why a capture is otherwise invisible to `--convert-linalg-to-cnm`.
//
//===----------------------------------------------------------------------===//

#include <cinm-mlir/Dialect/Cnm/Transforms/Passes.h>

#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/RegionUtils.h>

namespace mlir::cnm {

#define GEN_PASS_DEF_CNMISOLATELINALGCAPTURESPASS
#include <cinm-mlir/Dialect/Cnm/Transforms/Passes.h.inc>

} // namespace mlir::cnm

using namespace mlir;

namespace {

/// Materialize every value `op`'s region reads from outside itself as a new
/// scalar `ins` operand, so the op no longer captures anything. See the pass
/// description for why: `cnm.launch` clones a distributed op's region
/// verbatim into an `IsolatedFromAbove` body, and a capture that survives
/// that clone would dangle.
///
/// A capture becomes a rank-0 tensor rather than staying a bare scalar
/// operand, so that everything downstream that reads an operand's type as a
/// `ShapedType` -- which is everywhere in `--convert-linalg-to-cnm`, since an
/// operand's rank is what a projected-permutation indexing map is checked
/// against -- keeps working unchanged: rank 0 there just means "indexed by
/// none of the loop dimensions", which is exactly what a capture is.
static LogicalResult isolateCaptures(RewriterBase &rewriter,
                                     linalg::GenericOp op) {
  SetVector<Value> captures;
  getUsedValuesDefinedAbove(op.getRegion(), captures);
  if (captures.empty())
    return success();

  for (Value captured : captures)
    if (isa<ShapedType>(captured.getType()))
      return op->emitWarning(
                 "captures a shaped value defined outside its region (")
             << captured
             << "); only scalar captures can be turned into operands "
                "automatically. Skipping.";

  Location loc = op.getLoc();
  Block &body = op.getRegion().front();
  // New block arguments go right before the first `outs` argument: linalg's
  // convention is that block arguments mirror operand order, `ins` before
  // `outs`, and every capture becomes an `ins` operand.
  unsigned insertPos = op.getNumDpsInputs();
  AffineMap scalarMap =
      AffineMap::get(op.getNumLoops(), 0, {}, op.getContext());

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  for (Value captured : captures) {
    auto asTensor = tensor::FromElementsOp::create(
        rewriter, loc, RankedTensorType::get({}, captured.getType()),
        ValueRange{captured});
    maps.insert(maps.begin() + insertPos, scalarMap);
    BlockArgument arg = body.insertArgument(insertPos, captured.getType(), loc);
    rewriter.modifyOpInPlace(op, [&] {
      op.getInputsMutable().append(asTensor.getResult());
      op.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(maps));
    });
    // Scoped to the body: `captured` may be used elsewhere too (by another
    // op capturing the same value, or outside any linalg op at all), and
    // those uses must not see the new block argument.
    rewriter.replaceUsesWithIf(captured, arg, [&](OpOperand &use) {
      return use.getOwner()->getBlock() == &body;
    });
    ++insertPos;
  }
  return success();
}

struct CnmIsolateLinalgCapturesPass
    : public cnm::impl::CnmIsolateLinalgCapturesPassBase<
          CnmIsolateLinalgCapturesPass> {
  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    auto result = getOperation()->walk([&](linalg::GenericOp op) {
      (void)isolateCaptures(rewriter, op);
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace
