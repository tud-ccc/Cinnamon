#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/IR/TilingInterface.h"
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h"

#include <cstdint>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::cinm {

#define GEN_PASS_DEF_CINMTILINGPASS
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h.inc"

struct CinmApplyTilingInterfacePattern
    : public OpInterfaceConversionPattern<cinm::CinmTilingInterface> {

  CinmApplyTilingInterfacePattern(MLIRContext *context)
      : OpInterfaceConversionPattern<cinm::CinmTilingInterface>(context, 1) {
    setHasBoundedRewriteRecursion();
  }

  LogicalResult
  matchAndRewrite(cinm::CinmTilingInterface op, ArrayRef<Value>,
                  ConversionPatternRewriter &rewriter) const override {
    auto tileSizesAttr =
        op->getAttrOfType<DenseI64ArrayAttr>(CinmDialect::TILING_FACTORS_NAME);
    if (!tileSizesAttr) {
      // Should not be called bc op is illegal
      return failure();
    }

    SmallVector<Value> results;
    auto diag =
        op.convertToTiledOps(rewriter, tileSizesAttr.asArrayRef(), results);
    auto result = std::move(diag).checkAndReport();
    if (succeeded(result))
      rewriter.replaceOp(op, results);
    return result;
  }
};

struct CinmTilingPass : public impl::CinmTilingPassBase<CinmTilingPass> {
  using Base::Base;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<CinmApplyTilingInterfacePattern>(&getContext());

    ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal([](Operation *op) {
      if (auto tileable = llvm::dyn_cast_or_null<cinm::CinmTilingInterface>(op))
        return !tileable->hasAttr(cinm::CinmDialect::TILING_FACTORS_NAME);
      return true;
    });

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir::cinm
