#include "cinm-mlir/Conversion/LinalgToCinm/Im2ColToMatmul.h"

#include "cinm-mlir/Dialect/Cim/IR/CimDialect.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmDialect.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {

#define GEN_PASS_DEF_IM2COLTOMATMUL
#include "cinm-mlir/Conversion/CinmPasses.h.inc"
} // namespace mlir
using namespace mlir;

namespace {

using ReassociationIndices = SmallVector<int64_t, 2>;

static bool hasAllOneValues(DenseIntElementsAttr attr) {
  if (!attr)
    return false;
  return llvm::all_of(attr.getValues<int64_t>(),
                      [](int64_t v) { return v == 1; });
}

static Value transposeTensor(PatternRewriter &rewriter, Location loc,
                             Value value, ArrayRef<int64_t> permutation) {
  auto type = cast<RankedTensorType>(value.getType());
  SmallVector<int64_t> resultShape;
  resultShape.reserve(permutation.size());
  for (int64_t idx : permutation)
    resultShape.push_back(type.getShape()[idx]);

  Value empty = tensor::EmptyOp::create(rewriter, loc, resultShape,
                                        type.getElementType());

  SmallVector<AffineExpr> exprs;
  exprs.reserve(permutation.size());
  for (int64_t idx : permutation)
    exprs.push_back(rewriter.getAffineDimExpr(idx));

  auto inputMap =
      AffineMap::get(permutation.size(), 0, exprs, rewriter.getContext());
  SmallVector<AffineMap> maps = {
      inversePermutation(inputMap),
      AffineMap::getMultiDimIdentityMap(permutation.size(),
                                        rewriter.getContext())};

  SmallVector<utils::IteratorType> iteratorTypes(permutation.size(),
                                                 utils::IteratorType::parallel);

  auto generic = linalg::GenericOp::create(
      rewriter, loc, empty.getType(), ValueRange{value}, ValueRange{empty},
      maps, iteratorTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        linalg::YieldOp::create(nestedBuilder, nestedLoc, args[0]);
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

    Value inputNHWC =
        transposeTensor(rewriter, loc, op.getInputs()[0], {0, 2, 3, 1});
    Value filterHWC =
        transposeTensor(rewriter, loc, op.getInputs()[1], {1, 2, 0});

    ArrayRef<int64_t> outShape = resultType.getShape();
    SmallVector<int64_t> nhwcShape = {outShape[0], outShape[2], outShape[3],
                                      outShape[1]};
    auto nhwcType =
        RankedTensorType::get(nhwcShape, resultType.getElementType());
    Value init = tensor::EmptyOp::create(rewriter, loc, nhwcShape,
                                         nhwcType.getElementType());

    auto conv = linalg::DepthwiseConv2DNhwcHwcOp::create(
        rewriter, loc, nhwcType, ValueRange{inputNHWC, filterHWC},
        ValueRange{init}, op.getStridesAttr(), op.getDilationsAttr());

    Value resultNHWC = conv.getResult(0);
    Value resultNCHW = transposeTensor(rewriter, loc, resultNHWC, {0, 3, 1, 2});

    rewriter.replaceOp(op, resultNCHW);
    return success();
  }
};

struct GenericIm2ColMatmulToBatchMatmul
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1)
      return failure();
    if (op.getNumLoops() != 4)
      return failure();

    SmallVector<utils::IteratorType> iters = op.getIteratorTypesArray();
    if (iters.size() != 4)
      return failure();
    if (iters[0] != utils::IteratorType::parallel ||
        iters[1] != utils::IteratorType::parallel ||
        iters[2] != utils::IteratorType::parallel ||
        iters[3] != utils::IteratorType::reduction)
      return failure();

    auto indexingMaps = op.getIndexingMapsArray();
    if (indexingMaps.size() != 3)
      return failure();
    auto extractDims = [](AffineMap map) -> FailureOr<SmallVector<unsigned>> {
      SmallVector<unsigned> dims;
      dims.reserve(map.getNumResults());
      for (AffineExpr expr : map.getResults()) {
        auto dimExpr = dyn_cast<AffineDimExpr>(expr);
        if (!dimExpr)
          return failure();
        dims.push_back(dimExpr.getPosition());
      }
      return dims;
    };

    FailureOr<SmallVector<unsigned>> lhsDimsOr = extractDims(indexingMaps[0]);
    FailureOr<SmallVector<unsigned>> rhsDimsOr = extractDims(indexingMaps[1]);
    FailureOr<SmallVector<unsigned>> outDimsOr = extractDims(indexingMaps[2]);
    if (failed(lhsDimsOr) || failed(rhsDimsOr) || failed(outDimsOr))
      return failure();

    SmallVector<unsigned> lhsDims = *lhsDimsOr;
    SmallVector<unsigned> rhsDims = *rhsDimsOr;
    SmallVector<unsigned> outDims = *outDimsOr;

    if (lhsDims != SmallVector<unsigned>{1, 3})
      return failure();
    if (rhsDims != SmallVector<unsigned>{0, 3, 2})
      return failure();
    if (outDims != SmallVector<unsigned>{0, 1, 2})
      return failure();

    Block &body = op.getRegion().front();
    auto yield = cast<linalg::YieldOp>(body.getTerminator());
    if (yield.getNumOperands() != 1)
      return failure();

    Value mulResult;
    if (auto addf = yield.getOperand(0).getDefiningOp<arith::AddFOp>()) {
      if (!addf)
        return failure();
      if (!llvm::is_contained(addf->getOperands(), body.getArgument(2)))
        return failure();
      mulResult = addf.getOperand(0);
      if (mulResult == body.getArgument(2))
        mulResult = addf.getOperand(1);
      if (!mulResult || mulResult == body.getArgument(2))
        return failure();
      auto mulf = mulResult.getDefiningOp<arith::MulFOp>();
      if (!mulf)
        return failure();
      if (!((mulf.getLhs() == body.getArgument(0) &&
             mulf.getRhs() == body.getArgument(1)) ||
            (mulf.getLhs() == body.getArgument(1) &&
             mulf.getRhs() == body.getArgument(0))))
        return failure();
    } else if (auto addi = yield.getOperand(0).getDefiningOp<arith::AddIOp>()) {
      if (!addi)
        return failure();
      if (!llvm::is_contained(addi->getOperands(), body.getArgument(2)))
        return failure();
      mulResult = addi.getOperand(0);
      if (mulResult == body.getArgument(2))
        mulResult = addi.getOperand(1);
      if (!mulResult || mulResult == body.getArgument(2))
        return failure();
      auto muli = mulResult.getDefiningOp<arith::MulIOp>();
      if (!muli)
        return failure();
      if (!((muli.getLhs() == body.getArgument(0) &&
             muli.getRhs() == body.getArgument(1)) ||
            (muli.getLhs() == body.getArgument(1) &&
             muli.getRhs() == body.getArgument(0))))
        return failure();
    } else {
      return failure();
    }

    Value lhs = op.getDpsInputs()[0];
    Value rhs = op.getDpsInputs()[1];
    Value init = op.getDpsInits()[0];

    auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
    auto initType = dyn_cast<RankedTensorType>(init.getType());
    if (!lhsType || !rhsType || !initType)
      return failure();

    if (rhsType.getRank() != 3 || initType.getRank() != 3)
      return failure();

    Location loc = op.getLoc();

    auto dimsCompatible = [](int64_t a, int64_t b) {
      return ShapedType::isDynamic(a) || ShapedType::isDynamic(b) || a == b;
    };

    ArrayRef<int64_t> rhsShape = rhsType.getShape();
    ArrayRef<int64_t> initShape = initType.getShape();

    if (!dimsCompatible(rhsShape[0], initShape[0]) ||
        !dimsCompatible(rhsShape[2], initShape[2]))
      return failure();

    int64_t batchSize = rhsShape[0];
    int64_t kSize = rhsShape[1];
    int64_t nSize = rhsShape[2];

    if (!dimsCompatible(initShape[1], lhsType.getDimSize(0)))
      return failure();
    if (!dimsCompatible(kSize, lhsType.getDimSize(lhsType.getRank() - 1)))
      return failure();

    if (!dimsCompatible(initShape[2], nSize))
      return failure();

    auto elemType = lhsType.getElementType();
    if (rhsType.getElementType() != elemType ||
        initType.getElementType() != elemType)
      return failure();

    Value lhsExpanded = lhs;
    if (lhsType.getRank() == 2) {
      if (!(batchSize == 1 || batchSize == ShapedType::kDynamic))
        return failure();
      SmallVector<int64_t, 3> shape = {1, lhsType.getDimSize(0),
                                       lhsType.getDimSize(1)};
      auto expandedType = RankedTensorType::get(shape, elemType);
      SmallVector<ReassociationIndices, 2> reassoc;
      reassoc.push_back(ReassociationIndices{0, 1});
      reassoc.push_back(ReassociationIndices{2});
      lhsExpanded = tensor::ExpandShapeOp::create(rewriter, loc, expandedType,
                                                  lhs, reassoc);
    } else if (lhsType.getRank() != 3) {
      return failure();
    }

    auto batchMatmul = linalg::BatchMatmulOp::create(
        rewriter, loc, ValueRange{lhsExpanded, rhs}, ValueRange{init});

    rewriter.replaceOp(op, batchMatmul->getResults());
    return success();
  }
};

struct Im2ColToMatmulPass
    : public impl::Im2ColToMatmulBase<Im2ColToMatmulPass> {
  using Base = Im2ColToMatmulBase<Im2ColToMatmulPass>;
  using Base::Base;

  void runOnOperation() override {
    Operation *func = getOperation();
    RewritePatternSet patterns(func->getContext());
    cinm::populateIm2ColToMatmulPatterns(patterns, func->getContext());
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
  patterns.add<ConvertDepthwiseConv2DNchwChw, GenericIm2ColMatmulToBatchMatmul>(
      context);
  linalg::populateConvertConv2DToImg2ColPatterns(patterns);
}
