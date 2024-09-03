#include "cinm-mlir/Dialect/Cim/IR/CimDialect.h"
#include "cinm-mlir/Dialect/Cim/Transforms/Passes.h"
#include <mlir/IR/PatternMatch.h>

namespace mlir::cim {

bool isCimOp(Operation *op) {
  return op->getName().getStringRef().starts_with("cim.op");
}

bool isCimFuture(Value v) { return v.getType().isa<cim::FutureType>(); }

bool isLegalOp(Operation *op) {
  return !isCimOp(op) ||
         !llvm::any_of(op->getOperands().drop_front(), isCimFuture);
}

bool isLegalBarrier(Operation *op) {
  return !llvm::isa<cim::BarrierOp>(op) || isCimFuture(op->getOperand(0));
}

void populateCimEraseRedundantBarriersPatterns(RewritePatternSet &patterns,
                                               MLIRContext *ctx) {
  patterns.insert<CimEraseRedundantBarriersPattern>(ctx);
}

struct CimEraseRedundantBarriersPattern
    : public OpRewritePattern<cim::BarrierOp> {

  using OpRewritePattern<cim::BarrierOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(cim::BarrierOp op,
                                PatternRewriter &rewriter) const override {

    if (isLegalBarrier(op))
      return failure();

    op.getResult().replaceAllUsesWith(op.getOperand());
    rewriter.eraseOp(op);

    return success();
  }
};

} // namespace mlir::cim