#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/Interfaces/TilingInterface.h"
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

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_DEF_SOFTMAXTOCINMPASS
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

struct SoftmaxToCinmPattern : public OpConversionPattern<linalg::SoftmaxOp> {
  using OpConversionPattern<linalg::SoftmaxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::SoftmaxOp op,
                  OpConversionPattern<linalg::SoftmaxOp>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const auto loc = op.getLoc();
    const auto input = op.getInput();
    const ShapedType inputType = input.getType();
    cinm::ComputeOp computeOp =
        rewriter.replaceOpWithNewOp<cinm::ComputeOp>(op, op.getResultTypes());

    rewriter.setInsertionPointToEnd(&computeOp.getBody().emplaceBlock());
    const Value max = rewriter.create<cinm::ReduceOp>(
        loc, inputType.getElementType(), ReduceMethod::MAX, input, /*dims=*/0);
    const Value t = rewriter.create<cinm::SubsOp>(loc, input, max);
    const Value init = rewriter.create<tensor::EmptyOp>(
        loc, inputType.getShape(), inputType.getElementType());
    const Value e =
        rewriter
            .create<linalg::ExpOp>(
                loc,
                TypeRange{RankedTensorType::get(inputType.getShape(),
                                                inputType.getElementType())},
                ValueRange{t}, ValueRange{init})
            .getResult(0);
    const Value s = rewriter.create<cinm::ReduceOp>(
        loc, inputType.getElementType(), ReduceMethod::ADD, e, /*dims=*/0);
    const Value result = rewriter.create<cinm::DivsOp>(loc, e, s);
    rewriter.create<cinm::YieldOp>(loc, ValueRange{result});
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
