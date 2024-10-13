
#include "cinm-mlir/Conversion/CinmPasses.h"
#include "cinm-mlir/Dialect/Cim/IR/CimBase.h"
#include "cinm-mlir/Dialect/Cim/IR/CimOps.h"
#include "cinm-mlir/Dialect/Cim/IR/CimTypes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Utils/CinmUtils.h"

#include <algorithm>
#include <cassert>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

#include <vector>

using namespace mlir;
#define GEN_PASS_CLASSES
#include <cinm-mlir/Conversion/CinmPasses.h.inc>

namespace {

// Creates the specified type for a value with correct shape and element type
// Condition: The value must be shaped type
template <typename T> static T getShapedType(Value value) {
  auto shapedType = value.getType().cast<ShapedType>();
  return T::get(shapedType.getShape(), shapedType.getElementType());
}

// Prepares the cinm.compute operation for subsequent conversion passes
// by inserting cim.acquire and cim.release operations
struct ConvertCinmComputeToCim : public OpConversionPattern<cinm::ComputeOp> {

  ConvertCinmComputeToCim(MLIRContext *context)
      : OpConversionPattern<cinm::ComputeOp>(context, 2) {}

  // Used for marking the operation legal once the acquire operation is inserted
  // Actual replacement / erasure of the operation is done in a second pass
  // with the InlineCinmCompute pattern
  static bool preparedCinmComputeOp(Operation *op) {
    if (llvm::isa<cinm::ComputeOp>(op)) {
      auto computeOp = llvm::cast<cinm::ComputeOp>(op);
      return llvm::isa<cim::AcquireDeviceOp>(
          computeOp.getBody().front().front());
    }

    if (llvm::isa<cinm::CinmDialect>(op->getDialect()))
      return false;

    return true;
  }

  LogicalResult
  matchAndRewrite(cinm::ComputeOp op,
                  OpConversionPattern<cinm::ComputeOp>::OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.startOpModification(op);

    // NOTE: The following accesses to the operation inside the compute op
    // ares safe because there always has to be at least one nested operation

    // Acquire operations are needed for subsequent conversion passes
    // The operations are inserted at the beginning of the block
    rewriter.setInsertionPoint(&op.getBody().front().front());
    auto acquireDeviceOp = rewriter.create<cim::AcquireDeviceOp>(op->getLoc());
    auto acquireCrossbarOp = rewriter.create<cim::AcquireCrossbarOp>(
        op->getLoc(), acquireDeviceOp.getResult());

    // Release operation inserted at the end of the block
    rewriter.setInsertionPointAfter(&op.getBody().back().back());
    rewriter.create<cim::ReleaseCrossbarOp>(op->getLoc(),
                                            acquireCrossbarOp.getResult());
    rewriter.create<cim::ReleaseDeviceOp>(op->getLoc(),
                                          acquireDeviceOp.getResult());

    rewriter.finalizeOpModification(op);

    return success();
  }
};

// Convert cinm.yield to cim.barrier
// Each operand of cinm.yield is replaced by a cim.barrier operation
struct ConvertCinmYieldToCim : public OpConversionPattern<cinm::YieldOp> {

  using OpConversionPattern<cinm::YieldOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cinm::YieldOp op,
                  OpConversionPattern<cinm::YieldOp>::OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto parentComputeOp = op->getParentOfType<cinm::ComputeOp>();

    for (size_t i = 0; i < op->getNumOperands(); i++) {
      auto result = parentComputeOp.getResult(i);
      auto operand = op.getOperand(i);
      auto barrierOp = rewriter.create<cim::BarrierOp>(
          op.getLoc(), getShapedType<RankedTensorType>(operand), operand);

      result.replaceAllUsesWith(barrierOp.getResult());
    }

    rewriter.eraseOp(op);
    return success();
  }
};

// One to one replacement of operations
// The operation result of the replaced operation will be cim.future and a
// additional deviceId operand will be prepended
template <typename SourceOp, typename TargetOp>
struct CinmOneToOneNestedOpConversionPattern
    : public OpConversionPattern<SourceOp> {

  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, OpConversionPattern<SourceOp>::OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto parentComputeOp = op->template getParentOfType<cinm::ComputeOp>();
    auto &acquireCrossbarOp =
        *llvm::find_if(parentComputeOp.getBody().getOps(), [](Operation &op) {
          return llvm::isa<cim::AcquireCrossbarOp>(op);
        });

    auto resultType = getShapedType<cim::FutureType>(op.getResult());

    std::vector<Value> operands{};
    operands.reserve(op.getNumOperands() + 1);
    operands.push_back(acquireCrossbarOp.getResult(0));
    for (auto operand : op.getOperands())
      operands.push_back(operand);

    auto targetOp =
        rewriter.create<TargetOp>(op.getLoc(), resultType, operands);
    op.getResult().replaceAllUsesWith(targetOp.getResult());
    rewriter.eraseOp(op);
    return success();
  }
};

// Cleanup pattern. Just inlines all nested operations in cinm.compute
struct InlineCinmCompute : public OpConversionPattern<cinm::ComputeOp> {

  using OpConversionPattern<cinm::ComputeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cinm::ComputeOp op,
                  OpConversionPattern<cinm::ComputeOp>::OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Block *parentBlock = op->getBlock();
    auto insertionPoint = rewriter.getInsertionPoint();

    for (auto &nestedOp : llvm::make_early_inc_range(op.getBody().getOps())) {
      nestedOp.moveBefore(parentBlock, insertionPoint);
    }

    rewriter.eraseOp(op);

    return success();
  }
};

struct ConvertTiledCinmToCim
    : public ConvertTiledCinmToCimBase<ConvertTiledCinmToCim> {

  void runOnOperation() override {
    auto &ctx = getContext();

    // First pass:
    // First add cim.aquire and cim.release operations to cinm.compute
    // Then convert all nested operations to cim operations
    ConversionTarget firstPassTarget(ctx);
    firstPassTarget.addLegalDialect<cim::CimDialect>();
    firstPassTarget.markUnknownOpDynamicallyLegal(
        ConvertCinmComputeToCim::preparedCinmComputeOp);
    RewritePatternSet firstPassPatterns(&ctx);
    firstPassPatterns.insert<
        ConvertCinmComputeToCim,                                          //
        ConvertCinmYieldToCim,                                            //
        CinmOneToOneNestedOpConversionPattern<cinm::GemmOp, cim::GemmOp>, //
        CinmOneToOneNestedOpConversionPattern<cinm::GemvOp, cim::GemvOp>>(&ctx);

    if (failed(applyPartialConversion(getOperation(), firstPassTarget,
                                      std::move(firstPassPatterns))))
      signalPassFailure();

    // Second pass:
    // Inline cinm.compute operations
    RewritePatternSet secondPassPatterns(&ctx);
    ConversionTarget secondPassTarget(ctx);
    secondPassPatterns.insert<InlineCinmCompute>(&ctx);
    secondPassTarget.markUnknownOpDynamicallyLegal([](...) { return true; });
    secondPassTarget.addLegalDialect<cim::CimDialect>();
    secondPassTarget.addIllegalDialect<cinm::CinmDialect>();

    if (failed(applyPartialConversion(getOperation(), secondPassTarget,
                                      std::move(secondPassPatterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::cinm::createConvertTiledCinmToCimPass() {
  return std::make_unique<ConvertTiledCinmToCim>();
}

void mlir::cinm::registerCinmToCimPipeline() {
  // todo
}
