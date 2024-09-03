#include "cinm-mlir/Dialect/Cim/IR/CimBase.h"
#include "cinm-mlir/Dialect/Cim/IR/CimOps.h"
#include "cinm-mlir/Dialect/Cim/Transforms/Passes.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::cim {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_DEF_CIMSCHEDULEASAPPASS
#include "cinm-mlir/Dialect/Cim/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

struct CimScheduleAsapPattern : public RewritePattern {

  CimScheduleAsapPattern(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag{}, 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {

    if (!isCimOp(op) || isLegalOp(op))
      return failure();

    rewriter.startOpModification(op);

    // All operands except device id
    auto inputs = op->getOperands().drop_front();
    for (auto input : inputs) {
      if (!isCimFuture(input))
        continue;

      rewriter.setInsertionPointAfter(input.getDefiningOp());

      auto inputType = input.getType().cast<ShapedType>();
      auto tensorType = RankedTensorType::get(inputType.getShape(),
                                              inputType.getElementType());

      auto barrierOp =
          rewriter.create<cim::BarrierOp>(op->getLoc(), tensorType, input);

      input.replaceUsesWithIf(barrierOp.getResult(), [&](OpOperand &use) {
        return !use.getOwner()->isBeforeInBlock(op);
      });
    }

    rewriter.finalizeOpModification(op);

    return success();
  }
};

void populateCimScheduleAsapPatterns(RewritePatternSet &patterns,
                                     MLIRContext *context) {
  patterns.add<CimScheduleAsapPattern>(context);
  populateCimEraseRedundantBarriersPatterns(patterns, context);
}

bool cimScheduleAsapCheckDynamicallyLegal(Operation *op) {
  return isLegalOp(op) && isLegalBarrier(op);
}

struct CimScheduleAsapPass
    : public impl::CimScheduleAsapPassBase<CimScheduleAsapPass> {
  using Base::Base;

  void runOnOperation() final {
    auto &ctx = getContext();

    ConversionTarget target(ctx);
    target.markUnknownOpDynamicallyLegal(cimScheduleAsapCheckDynamicallyLegal);

    RewritePatternSet patterns(&ctx);
    populateCimScheduleAsapPatterns(patterns, &ctx);

    auto result =
        applyPartialConversion(getOperation(), target, std::move(patterns));

    if (failed(result))
      signalPassFailure();
  }
};

} // namespace mlir::cim
