
#include "cinm-mlir/Conversion/CinmFrontendPasses.h"

#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/IR/TilingInterface.h"
#include "cinm-mlir/Utils/CinmUtils.h"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/InliningUtils.h>
#include <torch-mlir/Dialect/Torch/IR/TorchOps.h>
#include <torch-mlir/Dialect/Torch/IR/TorchTypes.h>
#include <torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h>

using namespace mlir;
namespace mlir {
#define GEN_PASS_DEF_CONVERTTORCHTOCINM
#include <cinm-mlir/Conversion/CinmFrontendPasses.h.inc>
} // namespace mlir

namespace {

template <typename SourceOp, typename TargetOp>
struct ConvertTorchTensorOpToCinm : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, SourceOp::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto lhs = op.getOperand(0);
    auto lhsType = cast<torch::Torch::ValueTensorType>(lhs.getType());
    rewriter.setInsertionPointAfterValue(lhs);
    auto lhsConversionOp = torch::TorchConversion::ToBuiltinTensorOp::create(
        rewriter, op.getLoc(), lhsType.toBuiltinTensor(), lhs);

    auto rhs = op.getOperand(1);
    auto rhsType = cast<torch::Torch::ValueTensorType>(rhs.getType());
    // rewriter.setInsertionPointAfterValue(rhs);
    auto rhsConversionOp = torch::TorchConversion::ToBuiltinTensorOp::create(
        rewriter, op.getLoc(), rhsType.toBuiltinTensor(), rhs);

    auto result = op.getResult();
    auto resultType = cast<torch::Torch::ValueTensorType>(result.getType());

    rewriter.setInsertionPoint(op);
    auto cinmComputeBlockOp = cinm::ComputeBlockOp::create(
        rewriter, op.getLoc(), ValueRange{lhsConversionOp, rhsConversionOp},
        resultType.toBuiltinTensor());

    auto resultConversionOp =
        torch::TorchConversion::FromBuiltinTensorOp::create(
            rewriter, op.getLoc(), resultType, cinmComputeBlockOp.getResult(0));

    Block *computeBody = &cinmComputeBlockOp.getRegion().front();
    rewriter.setInsertionPointToStart(computeBody);

    auto targetOp =
        TargetOp::create(rewriter, op.getLoc(), computeBody->getArgument(0),
                         computeBody->getArgument(1));
    assert(targetOp.getResult() && "Is a tensor gemmlike");

    cinm::YieldOp::create(rewriter, op.getLoc(), targetOp.getResult());

    result.replaceAllUsesWith(resultConversionOp.getResult());
    rewriter.eraseOp(op);

    return success();
  }
};

struct ConvertTorchToCinm : public mlir::impl::ConvertTorchToCinmBase<ConvertTorchToCinm> {

  void runOnOperation() override {
    auto &ctx = getContext();

    RewritePatternSet patterns(&ctx);
    patterns.add<
        ConvertTorchTensorOpToCinm<torch::Torch::AtenMatmulOp, cinm::GemmOp>,
        ConvertTorchTensorOpToCinm<torch::Torch::AtenMmOp, cinm::GemmOp>,
        ConvertTorchTensorOpToCinm<torch::Torch::AtenMvOp, cinm::GemvOp>>(&ctx);

    ConversionTarget target(ctx);
    target.markUnknownOpDynamicallyLegal([](...) { return true; });
    target.addLegalDialect<cinm::CinmDialect>();
    target.addIllegalOp<torch::Torch::AtenMatmulOp>();
    target.addIllegalOp<torch::Torch::AtenMmOp>();
    target.addIllegalOp<torch::Torch::AtenMvOp>();

    if (applyPartialConversion(getOperation(), target, std::move(patterns))
            .failed())
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::cinm_frontend::createConvertTorchToCinmPass() {
  return std::make_unique<ConvertTorchToCinm>();
}
