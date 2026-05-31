//===- LowerMVMToStreaming.cpp - Lower MVM to streaming -------------===//
//
// Rewrites `alpine.mvm` into the streaming trio:
//   alpine.enqueue_vec → alpine.process(count = 1) → alpine.dequeue_vec
//
// This pass assumes the Alpine device contract is already satisfied, i.e. the
// involved buffers are memref<?xi8>. It does NOT insert quantization.
//
//===----------------------------------------------------------------------===//

#include "cinm-mlir/Dialect/Alpine/IR/AlpineOps.h"
#include "cinm-mlir/Dialect/Alpine/Transforms/Passes.h"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace mlir;
using namespace mlir::alpine;

//===- Generated pass def -------------------------------------------------===//

#define GEN_PASS_DEF_ALPINELOWERMVMTOSTREAMINGPASS
#include "cinm-mlir/Dialect/Alpine/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

namespace {

static bool isI8MemRef(Type t) {
  if (auto mem = dyn_cast<MemRefType>(t))
    return mem.getElementType().isInteger(8);
  return false;
}

struct LowerMVMToStreamingPattern : OpRewritePattern<MVMOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MVMOp op,
                                PatternRewriter &rewriter) const override {
    // Enforce i8 contract; leave op intact if not already satisfied.
    if (!isI8MemRef(op.getX().getType()) || !isI8MemRef(op.getY().getType()))
      return rewriter.notifyMatchFailure(
          op, "alpine.mvm expects i8 memrefs for x and y");

    Location loc = op.getLoc();
    Value tile = op.getTileId();

    // alpine.enqueue_vec %tile, %x
    rewriter.create<EnqueueVecOp>(loc, tile, op.getX());

    // alpine.process %tile { count = 1 }
    IntegerAttr one = rewriter.getI64IntegerAttr(1);
    rewriter.create<ProcessOp>(loc, tile,
                               /*activation=*/StringAttr(),
                               /*accumulate=*/rewriter.getBoolAttr(false),
                               /*count=*/one);

    // alpine.dequeue_vec %tile, %y
    rewriter.create<DequeueVecOp>(loc, tile, op.getY());

    rewriter.eraseOp(op);
    return success();
  }
};

// struct AlpineLowerMVMToStreamingPass
//     : public impl::AlpineLowerMVMToStreamingPassBase<
//           AlpineLowerMVMToStreamingPass> {
//   using Base::Base;

//   void runOnOperation() final {
//     RewritePatternSet patterns(&getContext());
//     patterns.add<LowerMVMToStreamingPattern>(&getContext());

//     GreedyRewriteConfig cfg;
//     cfg.useTopDownTraversal = true;
//     cfg.enableRegionSimplification = true;

//     if (failed(applyPatternsAndFoldGreedily(getOperation(),
//     std::move(patterns),
//                                             cfg))) {
//       signalPassFailure();
//     }
//   }
// };

} // namespace
