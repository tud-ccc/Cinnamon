#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/IR/TilingInterface.h"
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h"

#include <cmath>
#include <cstdint>
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>

namespace mlir::cinm {

#define GEN_PASS_DEF_SOFTMAXTOCINMPASS
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h.inc"

struct SoftmaxToCinmPattern : OpConversionPattern<linalg::SoftmaxOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::SoftmaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const auto loc = op.getLoc();
    const ShapedType inputType = op.getInput().getType();

    auto compute =
        rewriter.replaceOpWithNewOp<ComputeBlockOp>(op, adaptor.getOperands(), op.getResultTypes());
    Value innerInput = compute.getBodyArguments()[0];

    rewriter.setInsertionPointToEnd(&compute.getBody().emplaceBlock());
    const Value max = rewriter.create<cinm::ReduceOp>(loc, inputType.getElementType(),
                                                ReduceMethod::MAX, innerInput, 0);
    const Value t =
        rewriter
            .create<cinm::ElementwiseOp>(loc, ElementwiseKind::Sub, innerInput, max)
            .getResult();
    const SmallVector<Type, 1> types{RankedTensorType::get(
        inputType.getShape(), inputType.getElementType())};

    const Value e =
            cinm::ElementwiseOp::create(rewriter, loc, ElementwiseKind::Exp, t).getResult();
    const Value s = rewriter.create<cinm::ReduceOp>(loc, inputType.getElementType(),
                                              ReduceMethod::ADD, e, 0);
    const Value result =
        rewriter.create<cinm::ElementwiseOp>(loc, ElementwiseKind::Div, e, s)
            .getResult();
    rewriter.create<YieldOp>(loc, ValueRange{result});
    return success();
  }
};

struct SoftmaxToCinmPass
    : public impl::SoftmaxToCinmPassBase<SoftmaxToCinmPass> {
  using Base::Base;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.insert<SoftmaxToCinmPattern>(&getContext());
    ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal([](...) { return true; });
    target.addIllegalOp<linalg::SoftmaxOp>();

    if (applyPartialConversion(getOperation(), target, std::move(patterns))
            .failed())
      signalPassFailure();
  }
};

} // namespace mlir::cinm
