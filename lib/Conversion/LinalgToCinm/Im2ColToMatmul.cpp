#include "cinm-mlir/Conversion/LinalgToCinm/Im2ColToMatmul.h"

#include "cinm-mlir/Dialect/Cim/IR/CimDialect.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmDialect.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "cinm-mlir/Conversion/CinmPasses.h.inc"

namespace {

static bool hasAllOneValues(DenseIntElementsAttr attr) {
  if (!attr)
    return false;
  return llvm::all_of(attr.getValues<int64_t>(), [](int64_t v) { return v == 1; });
}

static Value transposeTensor(PatternRewriter &rewriter, Location loc, Value value,
                             ArrayRef<int64_t> permutation) {
  auto type = cast<RankedTensorType>(value.getType());
  SmallVector<int64_t> resultShape;
  resultShape.reserve(permutation.size());
  for (int64_t idx : permutation)
    resultShape.push_back(type.getShape()[idx]);

  Value empty = rewriter.create<tensor::EmptyOp>(loc, resultShape, type.getElementType());

  SmallVector<AffineExpr> exprs;
  exprs.reserve(permutation.size());
  for (int64_t idx : permutation)
    exprs.push_back(rewriter.getAffineDimExpr(idx));

  auto inputMap = AffineMap::get(permutation.size(), 0, exprs, rewriter.getContext());
  SmallVector<AffineMap> maps = {
      inversePermutation(inputMap),
      AffineMap::getMultiDimIdentityMap(permutation.size(), rewriter.getContext())};

  SmallVector<utils::IteratorType> iteratorTypes(permutation.size(),
                                                 utils::IteratorType::parallel);

  auto generic = rewriter.create<linalg::GenericOp>(
      loc, empty.getType(), ValueRange{value}, ValueRange{empty}, maps,
      iteratorTypes, [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        nestedBuilder.create<linalg::YieldOp>(nestedLoc, args[0]);
      });

  return generic.getResult(0);
}

struct ConvertDepthwiseConv2DNchwChw
    : public OpRewritePattern<linalg::DepthwiseConv2DNchwChwOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::DepthwiseConv2DNchwChwOp op,
                                PatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getInputs()[0].getType());
    auto filterType = dyn_cast<RankedTensorType>(op.getInputs()[1].getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());

    if (!inputType || !filterType || !resultType)
      return failure();
    if (!inputType.hasStaticShape() || !filterType.hasStaticShape() ||
        !resultType.hasStaticShape())
      return failure();
    if (!hasAllOneValues(op.getDilations()))
      return failure();

    Location loc = op.getLoc();

    Value inputNHWC = transposeTensor(rewriter, loc, op.getInputs()[0], {0, 2, 3, 1});
    Value filterHWC = transposeTensor(rewriter, loc, op.getInputs()[1], {1, 2, 0});

    ArrayRef<int64_t> outShape = resultType.getShape();
    SmallVector<int64_t> nhwcShape = {outShape[0], outShape[2], outShape[3], outShape[1]};
    auto nhwcType = RankedTensorType::get(nhwcShape, resultType.getElementType());
    Value init = rewriter.create<tensor::EmptyOp>(loc, nhwcShape, nhwcType.getElementType());

    auto conv = rewriter.create<linalg::DepthwiseConv2DNhwcHwcOp>(
        loc, nhwcType, ValueRange{inputNHWC, filterHWC}, ValueRange{init},
        op.getStridesAttr(), op.getDilationsAttr());

    Value resultNHWC = conv.getResult(0);
    Value resultNCHW = transposeTensor(rewriter, loc, resultNHWC, {0, 3, 1, 2});

    rewriter.replaceOp(op, resultNCHW);
    return success();
  }
};

struct Im2ColToMatmulPass
    : public Im2ColToMatmulBase<Im2ColToMatmulPass> {
  using Base = Im2ColToMatmulBase<Im2ColToMatmulPass>;
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(func.getContext());
    cinm::populateIm2ColToMatmulPatterns(patterns, func.getContext());
    if (failed(applyPatternsGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::cinm::createIm2ColToMatmulPass() {
  return std::make_unique<Im2ColToMatmulPass>();
}

void mlir::cinm::populateIm2ColToMatmulPatterns(RewritePatternSet &patterns,
                                                MLIRContext *context) {
  patterns.add<ConvertDepthwiseConv2DNchwChw>(context);
  linalg::populateConvertConv2DToImg2ColPatterns(patterns);
}
