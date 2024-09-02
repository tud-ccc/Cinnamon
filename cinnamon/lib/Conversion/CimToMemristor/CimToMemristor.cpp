
#include "cinm-mlir/Conversion/CimToMemristor/CimToMemristor.h"
#include "cinm-mlir/Dialect/Cim/IR/CimBase.h"
#include "cinm-mlir/Dialect/Cim/IR/CimOps.h"
#include "cinm-mlir/Dialect/Cim/IR/CimTypes.h"
#include "cinm-mlir/Dialect/Memristor/IR/MemristorAttributes.h"
#include "cinm-mlir/Dialect/Memristor/IR/MemristorBase.h"
#include "cinm-mlir/Dialect/Memristor/IR/MemristorOps.h"
#include "cinm-mlir/Dialect/Memristor/IR/MemristorTypes.h"

#include <cstdio>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace mlir;
#define GEN_PASS_CLASSES
#include <cinm-mlir/Conversion/CimPasses.h.inc>

namespace {

template <typename CimOp, typename MemristorOp>
struct ConvertCimOpToMemristor : OpConversionPattern<CimOp> {

  using OpConversionPattern<CimOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CimOp op, OpConversionPattern<CimOp>::OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto tileId = op.getOperand(0);
    auto resultShape = op.getResult().getType().template cast<ShapedType>();

    auto resultAllocOp = rewriter.create<bufferization::AllocTensorOp>(
        op.getLoc(),
        RankedTensorType::get(resultShape.getShape(),
                              resultShape.getElementType()),
        ValueRange{});

    auto createBufferizeOp = [&](Value value) {
      auto shapedType = value.getType().cast<ShapedType>();
      return rewriter.create<bufferization::ToMemrefOp>(
          op.getLoc(),
          MemRefType::get(shapedType.getShape(), shapedType.getElementType()),
          value);
    };

    auto aBufferizeOp = createBufferizeOp(op.getOperand(2));
    auto bBufferizeOp = createBufferizeOp(op.getOperand(1));
    auto resultBufferizeOp = createBufferizeOp(resultAllocOp.getResult());

    rewriter.create<memristor::WriteToCrossbarOp>(op.getLoc(), tileId,
                                                  bBufferizeOp.getResult());

    rewriter.create<MemristorOp>(op->getLoc(), tileId, aBufferizeOp.getResult(),
                                 resultBufferizeOp.getResult());

    op.getResult().replaceAllUsesWith(resultAllocOp.getResult());
    rewriter.eraseOp(op);

    return success();
  }
};

struct ConvertCimAcquireToMemristor : OpConversionPattern<cim::AcquireOp> {
  using OpConversionPattern<cim::AcquireOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cim::AcquireOp op,
                  OpConversionPattern<cim::AcquireOp>::OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto tileIdConstantOp = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));

    op.getResult().replaceAllUsesWith(tileIdConstantOp.getResult());
    rewriter.eraseOp(op);

    return success();
  }
};

struct EraseCimRelease : OpConversionPattern<cim::ReleaseOp> {
  using OpConversionPattern<cim::ReleaseOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cim::ReleaseOp op,
                  OpConversionPattern<cim::ReleaseOp>::OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertCimBarrierToMemristor : OpConversionPattern<cim::BarrierOp> {
  using OpConversionPattern<cim::BarrierOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cim::BarrierOp op,
                  OpConversionPattern<cim::BarrierOp>::OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto *cimOp = op.getOperand().getDefiningOp();

    rewriter.create<memristor::BarrierOp>(op.getLoc(), cimOp->getOperand(0));
    op.getResult().replaceAllUsesWith(cimOp->getResult(0));
    rewriter.eraseOp(op);

    return success();
  }
};

struct ConvertCimToMemristor
    : public ConvertCimToMemristorPassBase<ConvertCimToMemristor> {

  void runOnOperation() override {
    auto &ctx = getContext();

    ConversionTarget target(ctx);
    target.markUnknownOpDynamicallyLegal([](...) { return true; });
    target.addLegalDialect<memristor::MemristorDialect>();
    target.addLegalDialect<bufferization::BufferizationDialect>();
    target.addIllegalOp<cim::BarrierOp>();

    RewritePatternSet firstPassPatterns(&ctx);
    firstPassPatterns.insert<ConvertCimBarrierToMemristor>(&ctx);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(firstPassPatterns))))
      signalPassFailure();

    target.addIllegalDialect<cim::CimDialect>();

    RewritePatternSet secondPassPatterns(&ctx);
    secondPassPatterns
        .insert<ConvertCimAcquireToMemristor, EraseCimRelease,
                ConvertCimOpToMemristor<cim::GemmOp, memristor::GemmOp>,
                ConvertCimOpToMemristor<cim::GemvOp, memristor::GevmOp>>(&ctx);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(secondPassPatterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::cim::createConvertCimToMemristorPass() {
  return std::make_unique<ConvertCimToMemristor>();
}
