#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
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

namespace mlir {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_DEF_CINMTILEGEMMOPPASS
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

struct CinmTileGemmOpPattern : public OpConversionPattern<cinm::GemmOp> {
    using OpConversionPattern<cinm::GemmOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(cinm::GemmOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
        rewriter.replaceOp(op, op.tile(OpBuilder(op), SmallVector<int64_t>{16, 16, 32, 8, 4}));
        return success();
    }
};

struct CinmTileGemmOpPass : public impl::CinmTileGemmOpPassBase<CinmTileGemmOpPass> {
    using Base::Base;

    void runOnOperation() final {
        LLVMTypeConverter typeConverter(&getContext());
        RewritePatternSet patterns(&getContext());
        patterns.add<CinmTileGemmOpPattern>(&typeConverter.getContext());

        ConversionTarget target(getContext());
        target.addIllegalOp<cinm::GemmOp>();

        target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });

        if (failed(applyFullConversion(getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace mlir
