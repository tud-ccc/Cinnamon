/// Convert Cim → Memristor

#include "cinm-mlir/Conversion/CimToMemristor/CimToMemristor.h"

#include "cinm-mlir/Dialect/Cim/IR/CimBase.h"
#include "cinm-mlir/Dialect/Cim/IR/CimOps.h"
#include "cinm-mlir/Dialect/Cim/IR/CimTypes.h"
#include "cinm-mlir/Dialect/Memristor/IR/MemristorAttributes.h"
#include "cinm-mlir/Dialect/Memristor/IR/MemristorBase.h"
#include "cinm-mlir/Dialect/Memristor/IR/MemristorOps.h"
#include "cinm-mlir/Dialect/Memristor/IR/MemristorTypes.h"

#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace mlir;

// Generate only this TU’s pass base into mlir::cim::impl.
namespace mlir::cim {
#define GEN_PASS_DEF_CONVERTCIMTOMEMRISTORPASS
#include "cinm-mlir/Conversion/CimPasses.h.inc"
} // namespace mlir::cim

namespace {

/// If `v` is a tensor, create bufferization.to_memref for the same shape/etype.
/// If it’s already a memref, pass it through.
static Value toMemrefLike(ConversionPatternRewriter &rewriter, Location loc,
                          Value v) {
  Type ty = v.getType();
  if (auto mr = dyn_cast<MemRefType>(ty))
    return v;
  auto t = cast<RankedTensorType>(ty);
  auto memTy = MemRefType::get(t.getShape(), t.getElementType());
  return rewriter.create<bufferization::ToBufferOp>(loc, memTy, v);
}

/// Replace any direct `cim.barrier` users of `cimRes` with `replacementTensor`,
/// and also insert a `memristor.barrier` using `tileId` at the barrier’s loc.
static void rewriteImmediateBarriers(Value cimRes, Value replacementTensor,
                                     Value tileId,
                                     ConversionPatternRewriter &rewriter) {
  SmallVector<Operation *> toErase;
  for (Operation *user : cimRes.getUsers()) {
    if (auto bar = dyn_cast<cim::BarrierOp>(user)) {
      rewriter.setInsertionPoint(bar);
      rewriter.create<memristor::BarrierOp>(bar.getLoc(), tileId);
      bar.getResult().replaceAllUsesWith(replacementTensor);
      toErase.push_back(bar);
    }
  }
  for (Operation *op : toErase)
    rewriter.eraseOp(op);
}

struct ConvertCimGemvToMemristor : OpConversionPattern<cim::GemvOp> {
  using OpConversionPattern<cim::GemvOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cim::GemvOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Identify matrix vs vector among (lhs, rhs).
    Value lhs = op.getOperand(1);
    Value rhs = op.getOperand(2);
    auto lhsShaped = cast<ShapedType>(lhs.getType());
    Value matVal = (lhsShaped.getRank() == 2) ? lhs : rhs;
    Value vecVal = (lhsShaped.getRank() == 1) ? lhs : rhs;

    auto matTy = cast<ShapedType>(matVal.getType());
    Type elemTy = matTy.getElementType();
    int64_t rows =
        (matTy.getRank() == 2) ? matTy.getDimSize(0) : ShapedType::kDynamic;

    // Allocate result tensor and create memref views.
    auto outTensorTy = RankedTensorType::get({rows}, elemTy);
    Value outTensor = rewriter
                          .create<bufferization::AllocTensorOp>(
                              loc, outTensorTy, ValueRange{})
                          .getResult();
    auto outMemTy = MemRefType::get({rows}, elemTy);
    Value Y =
        rewriter.create<bufferization::ToBufferOp>(loc, outMemTy, outTensor);

    Value W = toMemrefLike(rewriter, loc, matVal);
    Value X = toMemrefLike(rewriter, loc, vecVal);
    Value tileId = op.getOperand(0);

    // Program weights then run GEMV (vector input).
    rewriter.create<memristor::WriteToCrossbarOp>(loc, tileId, W);
    rewriter.create<memristor::GevmOp>(loc, tileId, X, Y);

    // Rewrite any immediate barriers and erase the CIM op.
    rewriteImmediateBarriers(op.getResult(), outTensor, tileId, rewriter);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertCimGemmToMemristor : OpConversionPattern<cim::GemmOp> {
  using OpConversionPattern<cim::GemmOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cim::GemmOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Identify A (MxK) and B (KxN).
    Value A = op.getOperand(1);
    Value B = op.getOperand(2);
    auto Aty = cast<ShapedType>(A.getType());
    auto Bty = cast<ShapedType>(B.getType());
    if (!(Aty.getRank() == 2 && Bty.getRank() == 2))
      return op.emitError("expected rank-2 operands for GEMM");

    int64_t M = Aty.getDimSize(0);
    int64_t N = Bty.getDimSize(1);
    Type elemTy = Aty.getElementType();

    // Allocate result tensor (MxN) and create memref views.
    auto outTensorTy = RankedTensorType::get({M, N}, elemTy);
    Value outTensor = rewriter
                          .create<bufferization::AllocTensorOp>(
                              loc, outTensorTy, ValueRange{})
                          .getResult();
    auto outMemTy = MemRefType::get({M, N}, elemTy);
    Value C =
        rewriter.create<bufferization::ToBufferOp>(loc, outMemTy, outTensor);

    Value Am = toMemrefLike(rewriter, loc, A);
    Value Bm = toMemrefLike(rewriter, loc, B);
    Value tileId = op.getOperand(0);

    rewriter.create<memristor::WriteToCrossbarOp>(loc, tileId, Bm);
    rewriter.create<memristor::GemmOp>(loc, tileId, Am, C);

    rewriteImmediateBarriers(op.getResult(), outTensor, tileId, rewriter);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertCimAcquireToMemristor
    : OpConversionPattern<cim::AcquireDeviceOp> {
  using OpConversionPattern<cim::AcquireDeviceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cim::AcquireDeviceOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value deviceId = op.getResult();
    int32_t nextTile = 0;
    for (Operation *user : llvm::make_early_inc_range(deviceId.getUsers())) {
      if (!isa<cim::AcquireCrossbarOp>(user))
        continue;
      auto c = rewriter.create<arith::ConstantOp>(
          user->getLoc(), rewriter.getI32Type(),
          rewriter.getI32IntegerAttr(nextTile++));
      user->getResult(0).replaceAllUsesWith(c.getResult());
      rewriter.eraseOp(user);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct EraseCimReleaseDevice : OpConversionPattern<cim::ReleaseDeviceOp> {
  using OpConversionPattern<cim::ReleaseDeviceOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cim::ReleaseDeviceOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct EraseCimReleaseCrossbar : OpConversionPattern<cim::ReleaseCrossbarOp> {
  using OpConversionPattern<cim::ReleaseCrossbarOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cim::ReleaseCrossbarOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertCimToMemristor
    : public mlir::cim::impl::ConvertCimToMemristorPassBase<
          ConvertCimToMemristor> {
  using Base =
      mlir::cim::impl::ConvertCimToMemristorPassBase<ConvertCimToMemristor>;
  using Base::Base;

  void runOnOperation() override {
    MLIRContext &ctx = getContext();

    ConversionTarget target(ctx);
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    // Make the whole CIM dialect illegal; we’ll remove all of it.
    target.addIllegalDialect<cim::CimDialect>();

    // Legal targets.
    target.addLegalDialect<memristor::MemristorDialect>();
    target.addLegalDialect<bufferization::BufferizationDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<tensor::TensorDialect>();

    RewritePatternSet patterns(&ctx);
    patterns.insert<ConvertCimAcquireToMemristor, //
                    EraseCimReleaseDevice, EraseCimReleaseCrossbar,
                    ConvertCimGemvToMemristor, ConvertCimGemmToMemristor>(&ctx);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::cim::createConvertCimToMemristorPass() {
  return std::make_unique<ConvertCimToMemristor>();
}
