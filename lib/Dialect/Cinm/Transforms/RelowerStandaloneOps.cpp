#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::cinm;

namespace mlir::cinm {
namespace {

static LogicalResult ensureTensorOperands(Operation *op, ValueRange values) {
  for (Value v : values) {
    if (!isa<RankedTensorType>(v.getType()))
      return op->emitError("expected ranked tensor operands/results");
  }
  return success();
}

static Value buildEmptyLike(IRRewriter &rewriter, Location loc,
                            RankedTensorType type, Value exemplar) {
  SmallVector<Value> dynDims;
  for (int64_t i = 0, e = type.getRank(); i < e; ++i)
    if (type.isDynamicDim(i))
      dynDims.push_back(tensor::DimOp::create(rewriter, loc, exemplar, i));
  return tensor::EmptyOp::create(rewriter, loc, type.getShape(),
                                          type.getElementType(), dynDims);
}

static FailureOr<Value> buildBinaryElementwise(
    IRRewriter &rewriter, Location loc, Value lhs, Value rhs,
    RankedTensorType resultType,
    function_ref<FailureOr<Value>(Value, Value)> emitCombine) {
  Value init = buildEmptyLike(rewriter, loc, resultType, lhs);

  AffineMap id = rewriter.getMultiDimIdentityMap(resultType.getRank());
  SmallVector<AffineMap> maps{id, id, id};
  SmallVector<Attribute> iterTypeAttrs;
  iterTypeAttrs.reserve(resultType.getRank());
  for (int64_t i = 0, e = resultType.getRank(); i < e; ++i)
    iterTypeAttrs.push_back(linalg::IteratorTypeAttr::get(
        rewriter.getContext(), mlir::utils::IteratorType::parallel));
  auto indexingMaps = rewriter.getAffineMapArrayAttr(maps);
  auto iteratorTypesAttr = rewriter.getArrayAttr(iterTypeAttrs);

  bool combineFailed = false;
  auto generic = linalg::GenericOp::create(rewriter, 
      loc, TypeRange{resultType}, ValueRange{lhs, rhs}, ValueRange{init},
      indexingMaps, iteratorTypesAttr, StringAttr(), StringAttr(),
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        FailureOr<Value> combined = emitCombine(args[0], args[1]);
        if (failed(combined)) {
          combineFailed = true;
          linalg::YieldOp::create(nestedBuilder, nestedLoc, args[2]);
          return;
        }
        linalg::YieldOp::create(nestedBuilder, nestedLoc, *combined);
      },
      ArrayRef<NamedAttribute>{});

  if (combineFailed)
    return failure();
  return generic.getResult(0);
}

static FailureOr<Value> lowerAddLikeOp(IRRewriter &rewriter,
                                       cinm::ElementwiseOp op) {
  auto resultType = cast<RankedTensorType>(op.getResult().getType());
  Type elemTy = resultType.getElementType();
  auto combine = [&](Value a, Value b) -> FailureOr<Value> {
    if (isa<FloatType>(elemTy))
      return arith::AddFOp::create(rewriter, op.getLoc(), a, b).getResult();
    if (isa<IntegerType>(elemTy))
      return arith::AddIOp::create(rewriter, op.getLoc(), a, b).getResult();
    return failure();
  };
  return buildBinaryElementwise(rewriter, op.getLoc(), op.getLhs(), op.getRhs(),
                                resultType, combine);
}

static FailureOr<Value> lowerSubLikeOp(IRRewriter &rewriter,
                                       cinm::ElementwiseOp op) {
  auto resultType = cast<RankedTensorType>(op.getResult().getType());
  Type elemTy = resultType.getElementType();
  auto combine = [&](Value a, Value b) -> FailureOr<Value> {
    if (isa<FloatType>(elemTy))
      return arith::SubFOp::create(rewriter, op.getLoc(), a, b).getResult();
    if (isa<IntegerType>(elemTy))
      return arith::SubIOp::create(rewriter, op.getLoc(), a, b).getResult();
    return failure();
  };
  return buildBinaryElementwise(rewriter, op.getLoc(), op.getLhs(), op.getRhs(),
                                resultType, combine);
}

struct CinmRelowerPass
    : PassWrapper<CinmRelowerPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CinmRelowerPass)

  StringRef getArgument() const final { return "cinm-relower"; }
  StringRef getDescription() const final {
    return "Lower CINM tensor ops outside compute blocks back to linalg";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect,
                    arith::ArithDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    IRRewriter rewriter(func.getContext());

    SmallVector<cinm::ElementwiseOp> adds;
    SmallVector<cinm::ElementwiseOp> subs;

    func.walk([&](Operation *op) {
      if (auto elementwiseOp = dyn_cast<cinm::ElementwiseOp>(op)) {
        if (!elementwiseOp->getParentOfType<cinm::ComputeBlockOp>()) {
          if (elementwiseOp.getKind() == cinm::ElementwiseKind::Add) {
            adds.push_back(elementwiseOp);
          } else if (elementwiseOp.getKind() == cinm::ElementwiseKind::Sub) {
            subs.push_back(elementwiseOp);
          }
        }
      }
    });

    for (cinm::ElementwiseOp add : adds) {
      if (failed(ensureTensorOperands(
              add.getOperation(),
              {add.getLhs(), add.getRhs(), add.getResult()}))) {
        signalPassFailure();
        return;
      }
      rewriter.setInsertionPoint(add);
      FailureOr<Value> lowered = lowerAddLikeOp(rewriter, add);
      if (failed(lowered)) {
        add.emitError("failed to relower cinm.op.add");
        signalPassFailure();
        return;
      }
      rewriter.replaceOp(add, *lowered);
    }

    for (cinm::ElementwiseOp sub : subs) {
      if (failed(ensureTensorOperands(
              sub.getOperation(),
              {sub.getLhs(), sub.getRhs(), sub.getResult()}))) {
        signalPassFailure();
        return;
      }
      rewriter.setInsertionPoint(sub);
      FailureOr<Value> lowered = lowerSubLikeOp(rewriter, sub);
      if (failed(lowered)) {
        sub.emitError("failed to relower cinm.op.sub");
        signalPassFailure();
        return;
      }
      rewriter.replaceOp(sub, *lowered);
    }
  }
};

} // namespace
} // namespace mlir::cinm

std::unique_ptr<mlir::Pass> mlir::cinm::createCinmRelowerPass() {
  return std::make_unique<CinmRelowerPass>();
}
