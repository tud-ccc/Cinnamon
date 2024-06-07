#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/Interfaces/TilingInterface.h"
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h"

#include <llvm/ADT/SmallVector.h>
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

namespace mlir {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_DEF_CINMTILINGPASS
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

struct CinmApplyTilingInterfacePattern
    : public OpInterfaceConversionPattern<cinm::CinmTilingInterface> {
  using OpInterfaceConversionPattern<
      cinm::CinmTilingInterface>::OpInterfaceConversionPattern;

  CinmApplyTilingInterfacePattern(MLIRContext *context,
                                  ArrayRef<int64_t> tileSizes,
                                  int64_t reductionTileSize)
      : OpInterfaceConversionPattern<cinm::CinmTilingInterface>(context, 1),
        tileSizes(tileSizes), reductionTileSize(reductionTileSize) {}

  LogicalResult
  matchAndRewrite(cinm::CinmTilingInterface op, ArrayRef<Value>,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(
        op, op.convertToTiledOps(rewriter, tileSizes, reductionTileSize));
    return success();
  }

private:
  SmallVector<int64_t> tileSizes;
  int64_t reductionTileSize;
};

struct CinmTilingPass : public impl::CinmTilingPassBase<CinmTilingPass> {
  using Base::Base;

  void runOnOperation() final {
    LLVMTypeConverter typeConverter(&getContext());
    RewritePatternSet patterns(&getContext());
    patterns.add<CinmApplyTilingInterfacePattern>(&typeConverter.getContext(),
                                                  SmallVector<int64_t>{16, 16},
                                                  reductionTileSize);

    ConversionTarget target(getContext());

    target.addDynamicallyLegalOp<cinm::AddOp>([&](cinm::AddOp op) {
      return op.getResult().getType().getNumElements() <= reductionTileSize;
    });
    target.markUnknownOpDynamicallyLegal([](Operation *op) {
      return dyn_cast_or_null<cinm::CinmTilingInterface>(op) == nullptr;
    });

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir
