#include "cinm-mlir/Conversion/CinmPasses.h"

#include "cinm-mlir/Dialect/Cim/IR/CimAttributes.h"
#include "cinm-mlir/Dialect/Cim/IR/CimBase.h"
#include "cinm-mlir/Dialect/Cim/IR/CimOps.h"
#include "cinm-mlir/Dialect/Cim/IR/CimTypes.h"

#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"

#include "cinm-mlir/Utils/CinmUtils.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/Casting.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace mlir;

#define GEN_PASS_CLASSES
#include "cinm-mlir/Conversion/CinmPasses.h.inc"

namespace {

static Value toMemrefLike(ConversionPatternRewriter &rewriter, Location loc,
                          Value v) {
  Type ty = v.getType();
  if (isa<MemRefType>(ty))
    return v;
  auto t = dyn_cast<RankedTensorType>(ty);
  assert(t && "expected memref or ranked tensor");
  auto memTy = MemRefType::get(t.getShape(), t.getElementType());
  return rewriter.create<bufferization::ToBufferOp>(loc, memTy, v);
}

static Value getCrossbarIdFromCompute(cinm::ComputeOp computeOp) {
  for (Operation &nested : computeOp.getBody().getOps())
    if (auto acq = dyn_cast<cim::AcquireCrossbarOp>(&nested))
      return acq.getResult();
  return {};
}

static inline cim::RoundingMode mapRounding(cinm::RoundingMode r) {
  switch (r) {
  case cinm::RoundingMode::Nearest:
    return cim::RoundingMode::Nearest;
  case cinm::RoundingMode::TowardsZero:
    return cim::RoundingMode::TowardsZero;
  }
  llvm_unreachable("unknown cinm::RoundingMode");
}

static inline cim::ActivationKind toCimActivation(cinm::ActivationKind k) {
  switch (k) {
  case cinm::ActivationKind::RELU:
    return cim::ActivationKind::RELU;
  case cinm::ActivationKind::SIGMOID:
    return cim::ActivationKind::SIGMOID;
  case cinm::ActivationKind::TANH:
    return cim::ActivationKind::TANH;
  case cinm::ActivationKind::GELU:
    return cim::ActivationKind::GELU;
  }
  llvm_unreachable("unsupported cinm::ActivationKind");
}

struct ConvertCinmComputeToCim : public OpConversionPattern<cinm::ComputeOp> {
  using OpConversionPattern::OpConversionPattern;

  static bool preparedCinmComputeOp(Operation *op) {
    if (isa<cinm::ComputeOp>(op)) {
      auto computeOp = cast<cinm::ComputeOp>(op);
      return !computeOp.getBody().empty() &&
             !computeOp.getBody().front().empty() &&
             isa<cim::AcquireDeviceOp>(computeOp.getBody().front().front());
    }
    if (!isa<cinm::CinmDialect>(op->getDialect()))
      return true;
    return false;
  }

  LogicalResult
  matchAndRewrite(cinm::ComputeOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.startOpModification(op);

    auto &entryBlock = op.getBody().front();
    rewriter.setInsertionPoint(&entryBlock.front());
    auto acquireDev = rewriter.create<cim::AcquireDeviceOp>(op.getLoc());
    SmallVector<NamedAttribute> xbAttrs;
    if (auto tiles = op->getAttrOfType<DenseI64ArrayAttr>("tileSizes")) {
      auto vals = tiles.asArrayRef();
      if (vals.size() >= 2) {
        xbAttrs.emplace_back(rewriter.getStringAttr("height"),
                             rewriter.getI64IntegerAttr(vals[0]));
        xbAttrs.emplace_back(rewriter.getStringAttr("width"),
                             rewriter.getI64IntegerAttr(vals[1]));
      }
    }
    auto acquireXB = rewriter.create<cim::AcquireCrossbarOp>(
        op.getLoc(), acquireDev.getResult(), xbAttrs);

    Operation &lastOp = op.getBody().back().back();
    rewriter.setInsertionPointAfter(&lastOp);
    rewriter.create<cim::ReleaseCrossbarOp>(op.getLoc(), acquireXB.getResult());
    rewriter.create<cim::ReleaseDeviceOp>(op.getLoc(), acquireDev.getResult());

    rewriter.finalizeOpModification(op);
    return success();
  }
};

struct ConvertCinmYieldInMemRefCompute
    : public OpConversionPattern<cinm::YieldOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cinm::YieldOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto compute = op->getParentOfType<cinm::ComputeOp>();
    if (!compute)
      return op.emitOpError("must be nested in cinm.compute_memref");

    Location loc = op.getLoc();

    for (unsigned i = 0; i < op->getNumOperands(); ++i) {
      Value operand = op.getOperand(i);
      Value result = compute.getResult(i);

      if (auto futTy = dyn_cast<cim::FutureType>(operand.getType())) {
        (void)futTy;
        auto resTy = dyn_cast<MemRefType>(result.getType());
        if (!resTy)
          return op.emitOpError() << "compute_memref result #" << i
                                  << " must be a memref when yielding a future";
        auto barrier = rewriter.create<cim::BarrierOp>(loc, resTy, operand);
        result.replaceAllUsesWith(barrier.getResult());
        continue;
      }

      Value forwarded = operand;
      if (result.getType() != forwarded.getType()) {
        if (isa<RankedTensorType>(forwarded.getType()) &&
            isa<MemRefType>(result.getType())) {
          forwarded = toMemrefLike(rewriter, loc, forwarded);
        }
      }
      if (result.getType() != forwarded.getType())
        return op.emitOpError()
               << "type mismatch: result type " << result.getType()
               << " does not match yielded value type " << forwarded.getType();

      result.replaceAllUsesWith(forwarded);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct LowerCinmActivate : public OpConversionPattern<cinm::ActivateOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cinm::ActivateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto compute = op->getParentOfType<cinm::ComputeOp>();
    if (!compute)
      return op.emitOpError("must be nested in cinm.compute_memref");

    Location loc = op.getLoc();

    if (adaptor.getOperands().size() != 2)
      return op.emitOpError("expected (src, out) operands");

    Value src = adaptor.getOperands()[0];
    Value out = adaptor.getOperands()[1];

    src = toMemrefLike(rewriter, loc, src);

    auto outTy = dyn_cast<MemRefType>(out.getType());
    if (!outTy)
      return op.emitOpError("`out` must be a memref");

    auto futTy = cim::FutureType::get(outTy.getShape(), outTy.getElementType());

    OperationState st(loc, cim::ActivateOp::getOperationName());
    st.addTypes(futTy);
    st.addOperands(src);
    st.addAttribute(
        "kind", cim::ActivationKindAttr::get(rewriter.getContext(),
                                             toCimActivation(op.getKind())));
    Operation *act = rewriter.create(st);
    Value fut = act->getResult(0);

    Value y = rewriter.create<cim::BarrierOp>(loc, outTy, fut).getResult();
    rewriter.create<memref::CopyOp>(loc, y, out);

    rewriter.eraseOp(op);
    return success();
  }
};

struct LowerCinmQuantize : public OpConversionPattern<cinm::QuantizeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cinm::QuantizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto compute = op->getParentOfType<cinm::ComputeOp>();
    if (!compute)
      return op.emitOpError("must be nested in cinm.compute_memref");
    Value xb = getCrossbarIdFromCompute(compute);
    if (!xb)
      return op.emitOpError("missing cim.acquire_crossbar in compute_memref");

    Location loc = op.getLoc();

    Value src = toMemrefLike(rewriter, loc, adaptor.getSrc());
    Value out = op.getResult();
    auto outTy = dyn_cast<MemRefType>(out.getType());
    if (!outTy)
      return op.emitOpError("`out` must be a memref");

    auto futTy = cim::FutureType::get(outTy.getShape(), outTy.getElementType());

    OperationState st(loc, cim::QuantizeOp::getOperationName());
    st.addTypes(futTy);
    st.addOperands({xb, src});
    st.addAttribute("scale", op.getScaleAttr());
    st.addAttribute("zeroPoint", op.getZeroPointAttr());
    if (auto axis = op.getAxisAttr())
      st.addAttribute("axis", axis);
    st.addAttribute("rounding",
                    cim::RoundingModeAttr::get(rewriter.getContext(),
                                               mapRounding(op.getRounding())));
    st.addAttribute("narrowRange", rewriter.getBoolAttr(op.getNarrowRange()));

    auto *qOp = rewriter.create(st);
    Value fut = qOp->getResult(0);

    Value y = rewriter.create<cim::BarrierOp>(loc, outTy, fut).getResult();
    rewriter.create<memref::CopyOp>(loc, y, out);

    rewriter.eraseOp(op);
    return success();
  }
};

struct LowerCinmDequantize : public OpConversionPattern<cinm::DequantizeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cinm::DequantizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto compute = op->getParentOfType<cinm::ComputeOp>();
    if (!compute)
      return op.emitOpError("must be nested in cinm.compute_memref");
    Value xb = getCrossbarIdFromCompute(compute);
    if (!xb)
      return op.emitOpError("missing cim.acquire_crossbar in compute_memref");

    Location loc = op.getLoc();

    Value src = toMemrefLike(rewriter, loc, adaptor.getSrc());
    Value out = op.getResult();
    auto outTy = dyn_cast<MemRefType>(out.getType());
    if (!outTy)
      return op.emitOpError("`out` must be a memref");

    auto futTy = cim::FutureType::get(outTy.getShape(), outTy.getElementType());

    OperationState st(loc, cim::DequantizeOp::getOperationName());
    st.addTypes(futTy);
    st.addOperands({xb, src});
    st.addAttribute("scale", op.getScaleAttr());
    st.addAttribute("zeroPoint", op.getZeroPointAttr());
    if (auto axis = op.getAxisAttr())
      st.addAttribute("axis", axis);

    auto *dqOp = rewriter.create(st);
    Value fut = dqOp->getResult(0);

    Value y = rewriter.create<cim::BarrierOp>(loc, outTy, fut).getResult();
    rewriter.create<memref::CopyOp>(loc, y, out);

    rewriter.eraseOp(op);
    return success();
  }
};

struct LowerCinmGemm : public OpConversionPattern<cinm::GemmOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cinm::GemmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto compute = op->getParentOfType<cinm::ComputeOp>();
    if (!compute)
      return op.emitOpError("must be nested in cinm.compute_memref");

    Value xb = getCrossbarIdFromCompute(compute);
    if (!xb)
      return op.emitOpError("missing cim.acquire_crossbar in compute_memref");

    Location loc = op.getLoc();

    Value A = toMemrefLike(rewriter, loc, op.getLhs());
    Value B = toMemrefLike(rewriter, loc, op.getRhs());
    Value C = op.getResult();

    auto CTy = dyn_cast<MemRefType>(C.getType());
    if (!CTy || CTy.getRank() != 2)
      return op.emitOpError("`out` must be rank-2 memref");

    auto futTy = cim::FutureType::get(CTy.getShape(), CTy.getElementType());

    auto f = rewriter.create<cim::GemmOp>(loc, futTy, ValueRange{xb, A, B});
    auto y = rewriter.create<cim::BarrierOp>(loc, CTy, f.getResult());
    rewriter.create<memref::CopyOp>(loc, y.getResult(), C);

    rewriter.eraseOp(op);
    return success();
  }
};

struct LowerCinmGemv : public OpConversionPattern<cinm::GemvOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cinm::GemvOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto compute = op->getParentOfType<cinm::ComputeOp>();
    if (!compute)
      return op.emitOpError("must be nested in cinm.compute_memref");

    Value xb = getCrossbarIdFromCompute(compute);
    if (!xb)
      return op.emitOpError("missing cim.acquire_crossbar in compute_memref");

    Location loc = op.getLoc();

    Value A = toMemrefLike(rewriter, loc, adaptor.getLhs());
    Value x = toMemrefLike(rewriter, loc, adaptor.getRhs());
    Value yOut = op.getResult();

    auto yTy = dyn_cast<MemRefType>(yOut.getType());
    if (!yTy || yTy.getRank() != 1)
      return op.emitOpError("`out` must be rank-1 memref");

    auto futTy = cim::FutureType::get(yTy.getShape(), yTy.getElementType());

    auto f = rewriter.create<cim::GemvOp>(loc, futTy, ValueRange{xb, A, x});
    auto y = rewriter.create<cim::BarrierOp>(loc, yTy, f.getResult());
    rewriter.create<memref::CopyOp>(loc, y.getResult(), yOut);

    rewriter.eraseOp(op);
    return success();
  }
};

struct LowerCinmAdd : public OpConversionPattern<cinm::ElementwiseOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cinm::ElementwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getKind() != cinm::ElementwiseKind::Add)
      return failure();

    auto compute = op->getParentOfType<cinm::ComputeOp>();
    if (!compute)
      return op.emitOpError("must be nested in cinm.compute_memref");

    Value xb = getCrossbarIdFromCompute(compute);
    if (!xb)
      return op.emitOpError("missing cim.acquire_crossbar in compute_memref");

    Location loc = op.getLoc();

    Value lhs = toMemrefLike(rewriter, loc, op.getLhs());
    Value rhs = toMemrefLike(rewriter, loc, op.getRhs());
    Value out = op.getOut();

    auto outTy = dyn_cast<MemRefType>(out.getType());
    if (!outTy)
      return op.emitOpError("`out` must be a memref");

    auto futTy = cim::FutureType::get(outTy.getShape(), outTy.getElementType());

    auto f = rewriter.create<cim::AddOp>(loc, futTy, ValueRange{xb, lhs, rhs});
    auto y = rewriter.create<cim::BarrierOp>(loc, outTy, f.getResult());
    rewriter.create<memref::CopyOp>(loc, y.getResult(), out);

    rewriter.eraseOp(op);
    return success();
  }
};

struct InlineCinmCompute : public OpConversionPattern<cinm::ComputeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cinm::ComputeOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Block *parentBlock = op->getBlock();
    auto insertionPoint = rewriter.getInsertionPoint();
    for (auto &nested : llvm::make_early_inc_range(op.getBody().getOps()))
      nested.moveBefore(parentBlock, insertionPoint);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertTiledCinmToCim
    : public ConvertTiledCinmToCimBase<ConvertTiledCinmToCim> {

  void runOnOperation() override {
    MLIRContext &ctx = getContext();

    {
      ConversionTarget target(ctx);
      target.addLegalDialect<cim::CimDialect>();
      target.addLegalDialect<bufferization::BufferizationDialect>();
      target.addLegalDialect<memref::MemRefDialect>();
      target.addLegalDialect<arith::ArithDialect>();
      target.addLegalDialect<func::FuncDialect>();
      target.addLegalDialect<tensor::TensorDialect>();

      target.markUnknownOpDynamicallyLegal(
          ConvertCinmComputeToCim::preparedCinmComputeOp);

      RewritePatternSet patterns(&ctx);
      patterns.insert<ConvertCinmComputeToCim, ConvertCinmYieldInMemRefCompute,
                      LowerCinmActivate, LowerCinmGemm, LowerCinmGemv,
                      LowerCinmAdd, LowerCinmQuantize, LowerCinmDequantize>(
          &ctx);

      if (failed(applyPartialConversion(getOperation(), target,
                                        std::move(patterns))))
        return signalPassFailure();
    }

    {
      ConversionTarget target(ctx);
      target.addLegalDialect<cim::CimDialect>();
      target.addLegalDialect<bufferization::BufferizationDialect>();
      target.addLegalDialect<memref::MemRefDialect>();
      target.addLegalDialect<arith::ArithDialect>();
      target.addLegalDialect<func::FuncDialect>();
      target.addLegalDialect<tensor::TensorDialect>();

      target.addIllegalDialect<cinm::CinmDialect>();

      RewritePatternSet patterns(&ctx);
      patterns.insert<InlineCinmCompute>(&ctx);

      if (failed(applyPartialConversion(getOperation(), target,
                                        std::move(patterns))))
        return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::cinm::createConvertTiledCinmToCimPass() {
  return std::make_unique<ConvertTiledCinmToCim>();
}

void mlir::cinm::registerCinmToCimPipeline() {}
