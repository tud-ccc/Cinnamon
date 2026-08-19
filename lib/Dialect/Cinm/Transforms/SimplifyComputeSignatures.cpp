//===- SimplifyComputeSignatures.cpp - Narrow compute op boundaries -------===//
//
// Drops passthrough results and unused arguments of cinm.compute /
// cinm.compute_block ops. These rewrites used to be canonicalization
// patterns, but they change the op's signature, and the accelerator search
// splices a winning trial's body back into the original op by zipping
// operands and results positionally -- so the trial pipelines, which run the
// canonicalizer many times, must never change a block's boundary. This pass
// carries the rewrites instead and is only run by pipelines that own the
// blocks they narrow.
//
//===----------------------------------------------------------------------===//

#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmWorkgroupTypeInterface.h"
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace mlir::cinm {

#define GEN_PASS_DEF_CINMSIMPLIFYCOMPUTESIGNATURESPASS
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h.inc"

namespace {

/// A result whose yielded value is defined outside the region is dropped;
/// its uses are redirected to the outer value directly.
struct ComputeOpSimplifyYield : OpRewritePattern<cinm::ComputeOp> {
  using OpRewritePattern<ComputeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(cinm::ComputeOp op,
                                PatternRewriter &rewriter) const override {

    auto &block = op.getBody().front();
    auto yield = cast<cinm::YieldOp>(block.getTerminator());
    SmallVector<Value> oldResults;
    SmallVector<Value> newYielded;
    oldResults.reserve(yield->getNumOperands());
    newYielded.reserve(yield->getNumOperands());
    for (auto [yielded, result] :
         llvm::zip(yield->getOperands(), op.getResults())) {
      // if yielded value defined outside of the compute block, remove it
      Operation *owner = yielded.getDefiningOp();
      if (!owner)
        owner = yielded.getParentBlock()->getParentOp();
      if (block.findAncestorOpInBlock(*owner)) {
        newYielded.push_back(yielded);
        oldResults.push_back(result);
      } else {
        rewriter.replaceAllUsesWith(result, yielded);
      }
    }
    if (newYielded.size() == yield->getNumOperands())
      return failure();

    rewriter.setInsertionPointAfter(op);
    auto newOp =
        ComputeOp::create(rewriter, op.getLoc(),
                          ValueTypeRange<ValueRange>(ValueRange(newYielded)));
    newOp->setAttrs(op->getAttrs());
    yield->setOperands(newYielded);
    newOp.getBody().takeBody(op.getBody());
    for (auto [old, newer] : llvm::zip(oldResults, newOp.getResults())) {
      rewriter.replaceAllUsesWith(old, newer);
    }
    rewriter.eraseOp(op);

    return success();
  }
};

/// A result that re-yields a block argument is dropped; its uses are
/// redirected to the operand behind that argument.
struct ComputeBlockOpSimplifyYield : OpRewritePattern<cinm::ComputeBlockOp> {
  using OpRewritePattern<ComputeBlockOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(cinm::ComputeBlockOp op,
                                PatternRewriter &rewriter) const override {

    auto &block = op.getBody().front();
    auto yield = cast<cinm::YieldOp>(block.getTerminator());
    SmallVector<Value> keptResults;
    SmallVector<Value> keptYielded;
    keptResults.reserve(yield->getNumOperands());
    keptYielded.reserve(yield->getNumOperands());
    for (auto [yielded, result] :
         llvm::zip(yield->getOperands(), op.getResults())) {
      // if yielded value defined outside of the compute block, remove it
      Operation *owner = yielded.getDefiningOp();
      if (!owner)
        owner = yielded.getParentBlock()->getParentOp();
      if (auto bbarg = llvm::dyn_cast_or_null<BlockArgument>(yielded);
          bbarg && bbarg.getOwner()->getParentOp() == op) {
        auto outer = op->getOperands()[bbarg.getArgNumber()];
        rewriter.replaceAllUsesWith(result, outer);
      } else {
        keptYielded.push_back(yielded);
        keptResults.push_back(result);
      }
    }
    if (keptYielded.size() == yield->getNumOperands())
      return failure();

    rewriter.setInsertionPointAfter(op);
    auto newOp = ComputeBlockOp::create(
        rewriter, op.getLoc(), op.getOperands(),
        ValueTypeRange<ValueRange>(ValueRange(keptYielded)));
    newOp->setAttrs(op->getAttrs());
    yield->setOperands(keptYielded);
    newOp.getBody().takeBody(op.getBody());
    for (auto [old, newer] : llvm::zip(keptResults, newOp.getResults())) {
      rewriter.replaceAllUsesWith(old, newer);
    }
    rewriter.eraseOp(op);

    return success();
  }
};

struct ComputeBlockOpDeleteUnusedArgs : OpRewritePattern<cinm::ComputeBlockOp> {
  using OpRewritePattern<ComputeBlockOp>::OpRewritePattern;

  /// An argument this pattern must not treat as dead however unused it is: a
  /// forwarded workgroup (WorkgroupTypeInterface) is a residency declaration
  /// consumed only when the block is LOWERED -- the graph level allocated a
  /// device set outside and handed it in, and nothing at the cinm level has
  /// any reason to reference it yet. Deleting it would silently demote the
  /// member back to allocating its own set.
  static bool isKept(BlockArgument arg) {
    return !arg.use_empty() || isa<WorkgroupTypeInterface>(arg.getType());
  }

  LogicalResult matchAndRewrite(cinm::ComputeBlockOp op,
                                PatternRewriter &rewriter) const override {

    SmallVector<Value> keptOperands;
    keptOperands.reserve(op->getNumOperands());
    for (auto [bbarg, opnd] : op.zipArgsWithOperands()) {
      if (isKept(bbarg)) {
        keptOperands.push_back(opnd);
      }
    }
    if (keptOperands.size() == op->getNumOperands())
      return failure();

    rewriter.modifyOpInPlace(op, [&]() {
      op->setOperands(std::move(keptOperands));
      op.getBody().front().eraseArguments(
          [](BlockArgument arg) { return !isKept(arg); });
    });

    return success();
  }
};

struct SimplifyComputeSignaturesPass
    : public impl::CinmSimplifyComputeSignaturesPassBase<
          SimplifyComputeSignaturesPass> {
  using Base::Base;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ComputeOpSimplifyYield, ComputeBlockOpSimplifyYield,
                 ComputeBlockOpDeleteUnusedArgs>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::cinm
