#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir::cinm {
namespace {

static Value buildZeroLike(OpBuilder &b, Location loc, Type elemTy) {
  if (auto ft = dyn_cast<FloatType>(elemTy))
    return b.create<arith::ConstantOp>(loc, b.getFloatAttr(ft, 0.0));
  if (auto it = dyn_cast<IntegerType>(elemTy))
    return b.create<arith::ConstantOp>(loc, b.getIntegerAttr(it, 0));
  return {};
}

static LogicalResult rewriteGemvMemRefOnce(cinm::GemvOp op, IRRewriter &b) {
  Location loc = op.getLoc();

  Value out = op.getOut();
  auto outTy = dyn_cast<MemRefType>(out.getType());
  if (!outTy || outTy.getRank() != 1)
    return op.emitOpError("out must be rank-1 memref");

  SmallVector<Value> dynDims;
  for (int64_t d = 0; d < outTy.getRank(); ++d)
    if (outTy.isDynamicDim(d)) {
      Value idx = b.create<arith::ConstantIndexOp>(loc, d);
      dynDims.push_back(b.create<memref::DimOp>(loc, out, idx));
    }

  auto tmpTy = MemRefType::get(outTy.getShape(), outTy.getElementType());
  Value tmp = b.create<memref::AllocOp>(loc, tmpTy, dynDims);

  Value zero = buildZeroLike(b, loc, outTy.getElementType());
  if (!zero)
    return op.emitOpError("unsupported element type for zero init");
  (void)b.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{tmp});

  b.create<cinm::GemvOp>(loc, Type(), op.getLhs(), op.getRhs(), Value(), tmp);

  b.create<cinm::ElementwiseOp>(loc, Type(), ElementwiseKind::Add, out, tmp,
                                out);

  b.create<memref::DeallocOp>(loc, tmp);
  op.erase();
  return success();
}

static LogicalResult rewriteGemmMemRefOnce(cinm::GemmOp op, IRRewriter &b) {
  Location loc = op.getLoc();

  Value out = op.getOut();
  auto outTy = dyn_cast<MemRefType>(out.getType());
  if (!outTy || outTy.getRank() != 2)
    return op.emitOpError("out must be rank-2 memref");

  SmallVector<Value> dynDims;
  for (int64_t d = 0; d < outTy.getRank(); ++d)
    if (outTy.isDynamicDim(d)) {
      Value idx = b.create<arith::ConstantIndexOp>(loc, d);
      dynDims.push_back(b.create<memref::DimOp>(loc, out, idx));
    }

  auto tmpTy = MemRefType::get(outTy.getShape(), outTy.getElementType());
  Value tmp = b.create<memref::AllocOp>(loc, tmpTy, dynDims);

  Value zero = buildZeroLike(b, loc, outTy.getElementType());
  if (!zero)
    return op.emitOpError("unsupported element type for zero init");
  (void)b.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{tmp});

  b.create<cinm::GemmOp>(loc, Type(), op.getLhs(), op.getRhs(), Value(), tmp);

  b.create<cinm::ElementwiseOp>(loc, Type(), ElementwiseKind::Add, out, tmp,
                                out);

  b.create<memref::DeallocOp>(loc, tmp);
  op.erase();
  return success();
}

static LogicalResult rewriteGemvTensorBias(cinm::GemvOp op, IRRewriter &b) {
  if (!op.getBias())
    return success();

  Location loc = op.getLoc();
  auto resTy = cast<RankedTensorType>(op.getResult().getType());

  auto pure = b.create<cinm::GemvOp>(loc, resTy, op.getLhs(), op.getRhs(),
                                     Value(), Value());

  auto sum =
      b.create<cinm::ElementwiseOp>(loc, resTy, ElementwiseKind::Add,
                                    pure.getResult(), op.getBias(), Value());

  b.replaceOp(op, sum.getResult());
  return success();
}

static LogicalResult rewriteGemmTensorBias(cinm::GemmOp op, IRRewriter &b) {
  if (!op.getBias())
    return success();

  Location loc = op.getLoc();
  auto resTy = cast<RankedTensorType>(op.getResult().getType());

  auto pure = b.create<cinm::GemmOp>(loc, resTy, op.getLhs(), op.getRhs(),
                                     Value(), Value());

  auto sum =
      b.create<cinm::ElementwiseOp>(loc, resTy, ElementwiseKind::Add,
                                    pure.getResult(), op.getBias(), Value());

  b.replaceOp(op, sum.getResult());
  return success();
}

struct DecomposeCinmAccumulationsPass
    : PassWrapper<DecomposeCinmAccumulationsPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DecomposeCinmAccumulationsPass)

  StringRef getArgument() const final { return "cinm-decompose-accum"; }
  StringRef getDescription() const final {
    return "Decompose accumulating CINM GEMV/GEMM into pure compute + add "
           "(supports memref and tensor forms; tensor stays "
           "bufferization-friendly)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, linalg::LinalgDialect,
                    arith::ArithDialect, tensor::TensorDialect,
                    func::FuncDialect, cinm::CinmDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    IRRewriter b(func.getContext());

    SmallVector<cinm::GemvOp, 8> gemvMR;
    SmallVector<cinm::GemmOp, 8> gemmMR;
    SmallVector<cinm::GemvOp, 8> gemvT;
    SmallVector<cinm::GemmOp, 8> gemmT;

    func.walk([&](Operation *op) {
      if (auto gemv = dyn_cast<cinm::GemvOp>(op)) {
        if (dyn_cast<mlir::MemRefType>(gemv.getLhs().getType())) {
          gemvMR.push_back(gemv);
        } else {
          gemvT.push_back(gemv);
        }
      } else if (auto gemm = dyn_cast<cinm::GemmOp>(op)) {
        if (dyn_cast<mlir::MemRefType>(gemm.getLhs().getType())) {
          gemmMR.push_back(gemm);
        } else {
          gemmT.push_back(gemm);
        }
      }
    });

    for (cinm::GemvOp gmv : gemvMR) {
      b.setInsertionPoint(gmv);
      if (failed(rewriteGemvMemRefOnce(gmv, b)))
        return signalPassFailure();
    }
    for (cinm::GemmOp gmm : gemmMR) {
      b.setInsertionPoint(gmm);
      if (failed(rewriteGemmMemRefOnce(gmm, b)))
        return signalPassFailure();
    }

    for (cinm::GemvOp gv : gemvT) {
      if (!gv.getBias())
        continue;
      b.setInsertionPoint(gv);
      if (failed(rewriteGemvTensorBias(gv, b)))
        return signalPassFailure();
    }
    for (cinm::GemmOp gm : gemmT) {
      if (!gm.getBias())
        continue;
      b.setInsertionPoint(gm);
      if (failed(rewriteGemmTensorBias(gm, b)))
        return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createCinmDecomposeAccumulationPass() {
  return std::make_unique<DecomposeCinmAccumulationsPass>();
}

void registerDecomposeCinmAccumulationPass() {
  PassRegistration<DecomposeCinmAccumulationsPass>(
      []() -> std::unique_ptr<mlir::Pass> {
        return std::make_unique<DecomposeCinmAccumulationsPass>();
      });
}

} // namespace mlir::cinm
