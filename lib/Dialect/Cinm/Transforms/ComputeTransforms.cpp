#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include <cinm-mlir/Dialect/Cinm/Transforms/CinmTransforms.h>
#include <cinm-mlir/Dialect/Cinm/Transforms/Passes.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/RegionUtils.h>

using namespace mlir;

cinm::ComputeBlockOp cinm::isolateComputeBlock(cinm::ComputeOp op,
                                          RewriterBase &rewriter) {

  Region &sourceRegion = op->getRegion(0);

  auto captured = mlir::makeRegionIsolatedFromAbove(
      rewriter, sourceRegion,
      [](Operation *op2) { return m_Constant().match(op2); });

  rewriter.setInsertionPoint(op);
  auto newCompute = cinm::ComputeBlockOp::create(rewriter, op->getLoc(), captured,
                                            op->getResultTypes());
  newCompute->setAttrs(op->getAttrs());

  newCompute->getRegion(0).takeBody(sourceRegion);

  rewriter.replaceOp(op, newCompute);
  return newCompute;
}

cinm::ComputeOp cinm::deisolateComputeBlock(cinm::ComputeBlockOp op,
                                                RewriterBase &rewriter) {

  Region &sourceRegion = op->getRegion(0);

  for (auto [bbarg, value] : op.zipArgsWithOperands()) {
    rewriter.replaceAllUsesWith(bbarg, value);
  }

  rewriter.setInsertionPoint(op);
  auto newCompute =
      cinm::ComputeOp::create(rewriter, op->getLoc(), op->getResultTypes());
  newCompute->setAttrs(op->getAttrs());

  newCompute->getRegion(0).takeBody(sourceRegion);

  rewriter.replaceOp(op, newCompute);
  return newCompute;
}

void cinm::unwrapComputeBlockOp(cinm::ComputeOp op, RewriterBase &rewriter) {
  rewriter.setInsertionPointAfter(op);
  IRMapping mapper;
  for (auto &toCopy : op.getBody().front().without_terminator()) {
    rewriter.clone(toCopy, mapper);
  }
  auto term = op.getBody().front().getTerminator();
  for (auto [result, termOperand] :
       llvm::zip(op->getResults(), term->getOperands())) {
    rewriter.replaceAllUsesWith(result, mapper.lookup(termOperand));
  }
  rewriter.eraseOp(op);
}

void cinm::unwrapComputeBlockOp(cinm::ComputeBlockOp op, RewriterBase &rewriter) {
  rewriter.setInsertionPointAfter(op);
  IRMapping mapper;
  for (auto [arg, opnd] : op.zipArgsWithOperands()) {
    mapper.map(arg, opnd);
  }
  for (auto &toCopy : op.getBody().front().without_terminator()) {
    rewriter.clone(toCopy, mapper);
  }
  auto term = op.getBody().front().getTerminator();
  for (auto [result, termOperand] :
       llvm::zip(op->getResults(), term->getOperands())) {
    rewriter.replaceAllUsesWith(result, mapper.lookup(termOperand));
  }
  rewriter.eraseOp(op);
}

using namespace mlir;

namespace mlir::cinm {
#define GEN_PASS_DEF_CINMISOLATECOMPUTEPASS
#define GEN_PASS_DEF_CINMUNWRAPCOMPUTEBLOCKSPASS
#define GEN_PASS_DEF_CINMDEISOLATECOMPUTEBLOCKS
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h.inc"
} // namespace mlir::cinm

namespace {

struct IsolateFlexComputePattern : OpRewritePattern<cinm::ComputeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(cinm::ComputeOp op,
                                PatternRewriter &rewriter) const override {
    cinm::isolateComputeBlock(op, rewriter);
    return llvm::success();
  }
};

struct IsolateComputePass
    : public mlir::cinm::impl::CinmIsolateComputePassBase<IsolateComputePass> {

  void runOnOperation() override {
    mlir::RewritePatternSet set(&getContext());
    set.add<IsolateFlexComputePattern>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(set))))
      signalPassFailure();
  }
};

struct DeisolateComputePattern : OpRewritePattern<cinm::ComputeBlockOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(cinm::ComputeBlockOp op,
                                PatternRewriter &rewriter) const override {
    cinm::deisolateComputeBlock(op, rewriter);
    return llvm::success();
  }
};
struct DeIsolateComputePass
    : public mlir::cinm::impl::CinmDeisolateComputeBlocksBase<
          DeIsolateComputePass> {

  void runOnOperation() override {
    mlir::RewritePatternSet set(&getContext());
    set.add<DeisolateComputePattern>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(set))))
      signalPassFailure();
  }
};

struct UnwrapComputeBlocks
    : public cinm::impl::CinmUnwrapComputeBlocksPassBase<UnwrapComputeBlocks> {

  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    getOperation()->walk([&](Operation *op) {
      if (auto compute = llvm::dyn_cast_or_null<cinm::ComputeBlockOp>(op)) {
        cinm::unwrapComputeBlockOp(compute, rewriter);
      } else if (auto compute =
                     llvm::dyn_cast_or_null<cinm::ComputeOp>(op)) {
        cinm::unwrapComputeBlockOp(compute, rewriter);
      }
    });
  }
};

} // namespace
