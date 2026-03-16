#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace mlir;

cinm::ComputeOp cinm::isolateComputeBlock(cinm::FlexComputeOp op,
                                          RewriterBase &rewriter) {

  Region &sourceRegion = op->getRegion(0);

  auto captured = mlir::makeRegionIsolatedFromAbove(
      rewriter, sourceRegion,
      [](Operation *op2) { return m_Constant().match(op2); });

  rewriter.setInsertionPoint(op);
  auto newCompute = cinm::ComputeOp::create(rewriter, op->getLoc(), captured,
                                            op->getResultTypes());
  newCompute->setAttrs(op->getAttrs());

  newCompute->getRegion(0).takeBody(sourceRegion);

  rewriter.replaceOp(op, newCompute);
  return newCompute;
}

namespace mlir::cinm {
#define GEN_PASS_DEF_CINMISOLATECOMPUTEPASS
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h.inc"
} // namespace mlir::cinm

namespace {

struct Pattern : OpRewritePattern<cinm::FlexComputeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(cinm::FlexComputeOp op,
                                PatternRewriter &rewriter) const override {
    cinm::isolateComputeBlock(op, rewriter);
    return llvm::success();
  }
};
struct IsolateComputePass
    : public mlir::cinm::impl::CinmIsolateComputePassBase<IsolateComputePass> {

  void runOnOperation() override {
    mlir::RewritePatternSet set(&getContext());
    set.add<Pattern>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(set))))
      signalPassFailure();
  }
};
} // namespace
