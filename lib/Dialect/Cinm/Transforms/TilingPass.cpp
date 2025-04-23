#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/Interfaces/TilingInterface.h"
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::cinm {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_DEF_CINMTILINGPASS
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

struct CinmApplyTilingInterfacePattern
    : public OpInterfaceConversionPattern<cinm::CinmTilingInterface> {

  CinmApplyTilingInterfacePattern(MLIRContext *context)
      : OpInterfaceConversionPattern<cinm::CinmTilingInterface>(context, 1) {
    setHasBoundedRewriteRecursion();
  }

  LogicalResult
  matchAndRewrite(cinm::CinmTilingInterface op, ArrayRef<Value>,
                  ConversionPatternRewriter &rewriter) const override {
    auto computeBlock = mlir::cinm::getEnclosingComputeBlock(op);
    auto tilingParms = cinm::TilingParameters::fromComputeBlock(computeBlock);
    auto result = op.convertToTiledOps(rewriter, tilingParms);
    if (succeeded(result)) {
      rewriter.replaceOp(op, *result);
      return success();
    } else {
      markOpAsNoTile(op);
    }
    return failure();
  }
};

struct CinmTilingPass : public impl::CinmTilingPassBase<CinmTilingPass> {
  using Base::Base;

  void runOnOperation() final {
    LLVMTypeConverter typeConverter(&getContext());
    RewritePatternSet patterns(&getContext());
    patterns.add<CinmApplyTilingInterfacePattern>(&typeConverter.getContext());

    ConversionTarget target(getContext());

    target.markUnknownOpDynamicallyLegal([](Operation *op) {
      if (auto tileable = llvm::dyn_cast_or_null<cinm::CinmTilingInterface>(op))
        return tileable->hasAttr(cinm::CinmDialect::NOTILE_NAME);
      return true;
    });

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir::cinm
