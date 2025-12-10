#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmUtils.h"
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::cinm;

namespace mlir::cinm {

#define GEN_PASS_DEF_CINMGEMMTOLOOPEDGEMVPASS
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h.inc"

namespace {

struct GemmToLoopedGemv final : OpRewritePattern<cinm::GemmOp> {
  using OpRewritePattern<cinm::GemmOp>::OpRewritePattern;

  explicit GemmToLoopedGemv(MLIRContext *ctx, int splitDim)
      : OpRewritePattern<cinm::GemmOp>(ctx), splitDim(splitDim) {}

  LogicalResult matchAndRewrite(cinm::GemmOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    cinm::ComputeOp parentCompute = op->getParentOfType<cinm::ComputeOp>();

    auto aTy = dyn_cast<RankedTensorType>(op.getLhs().getType());
    auto bTy = dyn_cast<RankedTensorType>(op.getRhs().getType());
    auto yTy = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!aTy || !bTy || !yTy || aTy.getRank() != 2 || bTy.getRank() != 2 ||
        yTy.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "expected ranked 2D tensors");

    const int64_t M = aTy.getDimSize(0);
    const int64_t K = aTy.getDimSize(1);
    const int64_t K2 = bTy.getDimSize(0);
    const int64_t N = bTy.getDimSize(1);
    if (ShapedType::isDynamic(M) || ShapedType::isDynamic(K) ||
        ShapedType::isDynamic(K2) || ShapedType::isDynamic(N))
      return rewriter.notifyMatchFailure(op, "dynamic dims not supported");
    if (K != K2)
      return rewriter.notifyMatchFailure(op, "inner dims must match (K)");

    Type elemTy = aTy.getElementType();
    if (elemTy != bTy.getElementType() || elemTy != yTy.getElementType())
      return rewriter.notifyMatchFailure(op, "element types must match");

    Value acc0 =
        rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{M, N}, elemTy);

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    SmallVector<int64_t, 2> staticOffsets{ShapedType::kDynamic,
                                          ShapedType::kDynamic};
    SmallVector<int64_t, 2> staticStrides{1, 1};

    SmallVector<Value> finals;

    if (splitDim == 2) {
      finals = createNestedAffineForLoops(
          rewriter, loc, ArrayRef<int64_t>{N}, ArrayRef<int64_t>{1},
          ValueRange{acc0},
          [&](OpBuilder &b, Location loc2, ValueRange ivs,
              ValueRange iters) -> SmallVector<Value> {
            Value j = ivs[0];

            SmallVector<int64_t, 2> staticSizesB{K, 1};
            auto bCol2DTy = RankedTensorType::get({K, 1}, elemTy);
            Value bCol2D = b.create<tensor::ExtractSliceOp>(
                loc2, bCol2DTy, op.getRhs(), ValueRange{c0, j}, ValueRange{},
                ValueRange{}, staticOffsets, staticSizesB, staticStrides);

            SmallVector<ReassociationIndices, 1> collapse{{0, 1}};
            auto xTy = RankedTensorType::get({K}, elemTy);
            Value x =
                b.create<tensor::CollapseShapeOp>(loc2, xTy, bCol2D, collapse);

            auto yColTy = RankedTensorType::get({M}, elemTy);
            Value yj = b.create<cinm::GemvOp>(loc2, yColTy, op.getLhs(), x,
                                              Value(), Value())
                           .getResult();

            SmallVector<ReassociationIndices, 1> expand{{0, 1}};
            auto yCol2DTy = RankedTensorType::get({M, 1}, elemTy);
            Value yCol2D =
                b.create<tensor::ExpandShapeOp>(loc2, yCol2DTy, yj, expand);

            SmallVector<int64_t, 2> staticSizesY{M, 1};
            Value accIn = iters.front();
            Value accOut = b.create<tensor::InsertSliceOp>(
                loc2, yCol2D, accIn, ValueRange{c0, j}, ValueRange{},
                ValueRange{}, staticOffsets, staticSizesY, staticStrides);

            return {accOut};
          });
    } else {
      Value btInit = rewriter.create<tensor::EmptyOp>(
          loc, ArrayRef<int64_t>{N, K}, elemTy);
      Value bTransposed =
          createNestedAffineForLoops(
              rewriter, loc, ArrayRef<int64_t>{N, K}, ArrayRef<int64_t>{1, 1},
              ValueRange{btInit},
              [&](OpBuilder &b, Location loc2, ValueRange ivs,
                  ValueRange iters) -> SmallVector<Value> {
                Value j = ivs[0];
                Value kVal = ivs[1];
                Value element = b.create<tensor::ExtractOp>(
                    loc2, op.getRhs(), ValueRange{kVal, j});
                Value updated = b.create<tensor::InsertOp>(
                    loc2, element, iters.front(), ValueRange{j, kVal});
                return {updated};
              })
              .front();

      finals = createNestedAffineForLoops(
          rewriter, loc, ArrayRef<int64_t>{M}, ArrayRef<int64_t>{1},
          ValueRange{acc0},
          [&](OpBuilder &b, Location loc2, ValueRange ivs,
              ValueRange iters) -> SmallVector<Value> {
            Value i = ivs[0];

            SmallVector<int64_t, 2> staticSizesA{1, K};
            auto aRow2DTy = RankedTensorType::get({1, K}, elemTy);
            Value aRow2D = b.create<tensor::ExtractSliceOp>(
                loc2, aRow2DTy, op.getLhs(), ValueRange{i, c0}, ValueRange{},
                ValueRange{}, staticOffsets, staticSizesA, staticStrides);

            SmallVector<ReassociationIndices, 1> collapse01{{0, 1}};
            auto aRowTy = RankedTensorType::get({K}, elemTy);
            Value aRow = b.create<tensor::CollapseShapeOp>(loc2, aRowTy, aRow2D,
                                                           collapse01);

            auto yRowTy = RankedTensorType::get({N}, elemTy);
            Value yRow = b.create<cinm::GemvOp>(loc2, yRowTy, bTransposed, aRow,
                                                Value(), Value())
                             .getResult();

            SmallVector<ReassociationIndices, 1> expand01{{0, 1}};
            auto yRow2DTy = RankedTensorType::get({1, N}, elemTy);
            Value yRow2D =
                b.create<tensor::ExpandShapeOp>(loc2, yRow2DTy, yRow, expand01);

            SmallVector<int64_t, 2> staticSizesY{1, N};
            Value accIn = iters.front();
            Value accOut = b.create<tensor::InsertSliceOp>(
                loc2, yRow2D, accIn, ValueRange{i, c0}, ValueRange{},
                ValueRange{}, staticOffsets, staticSizesY, staticStrides);

            return {accOut};
          });
    }

    rewriter.replaceOp(op, finals.front());

    if (parentCompute) {
      if (auto ts =
              parentCompute->getAttrOfType<DenseI64ArrayAttr>("tileSizes")) {
        auto arr = ts.asArrayRef();
        if (arr.size() == 3) {
          SmallVector<int64_t, 2> gemvTS =
              (splitDim == 2) ? SmallVector<int64_t, 2>{arr[0], arr[2]}
                              : SmallVector<int64_t, 2>{arr[1], arr[2]};
          rewriter.modifyOpInPlace(parentCompute, [&] {
            parentCompute->setAttr("tileSizes", rewriter.getDenseI64ArrayAttr(
                                                    ArrayRef<int64_t>(gemvTS)));
          });
        }
      }
    }

    return success();
  }

  int splitDim;
};

struct BatchGemmToLoopedGemv final : OpRewritePattern<cinm::BatchGemmOp> {
  using OpRewritePattern<cinm::BatchGemmOp>::OpRewritePattern;

  explicit BatchGemmToLoopedGemv(MLIRContext *ctx, int splitDim)
      : OpRewritePattern<cinm::BatchGemmOp>(ctx), splitDim(splitDim) {}

  LogicalResult matchAndRewrite(cinm::BatchGemmOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto aTy = dyn_cast<RankedTensorType>(op.getLeft().getType());
    auto bTy = dyn_cast<RankedTensorType>(op.getRight().getType());
    auto yTy = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!aTy || !bTy || !yTy || aTy.getRank() != 3 || bTy.getRank() != 3 ||
        yTy.getRank() != 3)
      return rewriter.notifyMatchFailure(op, "expected ranked 3D tensors");

    const int64_t B = aTy.getDimSize(0);
    const int64_t M = aTy.getDimSize(1);
    const int64_t K = aTy.getDimSize(2);
    const int64_t K2 = bTy.getDimSize(1);
    const int64_t N = bTy.getDimSize(2);
    if (ShapedType::isDynamic(B) || ShapedType::isDynamic(M) ||
        ShapedType::isDynamic(K) || ShapedType::isDynamic(K2) ||
        ShapedType::isDynamic(N))
      return rewriter.notifyMatchFailure(op, "dynamic dims not supported");
    if (K != K2)
      return rewriter.notifyMatchFailure(op, "inner dims must match (K)");

    Type elemTy = aTy.getElementType();
    if (elemTy != bTy.getElementType() || elemTy != yTy.getElementType())
      return rewriter.notifyMatchFailure(op, "element types must match");

    auto parentCompute = op->getParentOfType<cinm::ComputeOp>();

    Value bias = op.getBias();
    RankedTensorType biasTy;
    if (bias) {
      biasTy = dyn_cast<RankedTensorType>(bias.getType());
      if (!biasTy || biasTy.getRank() != 3)
        return rewriter.notifyMatchFailure(op, "bias must be ranked 3D tensor");
      if (biasTy.getDimSize(0) != B || biasTy.getDimSize(1) != M ||
          biasTy.getDimSize(2) != N)
        return rewriter.notifyMatchFailure(op, "bias dims must match (B,M,N)");
      if (biasTy.getElementType() != elemTy)
        return rewriter.notifyMatchFailure(op, "bias element type must match");
    }

    Value acc0 = rewriter.create<tensor::EmptyOp>(
        loc, ArrayRef<int64_t>{B, M, N}, elemTy);

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    SmallVector<Value> finals;

    if (splitDim == 2) {
      SmallVector<int64_t, 3> dynOffsets3(3, ShapedType::kDynamic);
      SmallVector<int64_t, 3> unitStrides3{1, 1, 1};

      SmallVector<int64_t, 3> staticSizesABatch{1, M, K};
      auto aBatchTy = RankedTensorType::get({1, M, K}, elemTy);
      SmallVector<ReassociationIndices, 2> collapseABatch{{0, 1}, {2}};
      auto aMatTy = RankedTensorType::get({M, K}, elemTy);

      SmallVector<int64_t, 3> staticSizesBCol{1, K, 1};
      auto bColTy = RankedTensorType::get({1, K, 1}, elemTy);
      SmallVector<ReassociationIndices, 1> collapseBCol{{0, 1, 2}};
      auto xTy = RankedTensorType::get({K}, elemTy);

      SmallVector<ReassociationIndices, 1> expandY{{0, 1, 2}};
      auto yExpandedTy = RankedTensorType::get({1, M, 1}, elemTy);
      SmallVector<int64_t, 3> staticSizesY{1, M, 1};
      auto yVecTy = RankedTensorType::get({M}, elemTy);

      SmallVector<int64_t, 3> staticSizesBias{1, M, 1};
      auto biasColTy = RankedTensorType::get({1, M, 1}, elemTy);
      SmallVector<ReassociationIndices, 1> collapseBias{{0, 1, 2}};
      auto biasVecTy = RankedTensorType::get({M}, elemTy);

      finals = createNestedAffineForLoops(
          rewriter, loc, ArrayRef<int64_t>{B, N}, ArrayRef<int64_t>{1, 1},
          ValueRange{acc0},
          [&](OpBuilder &b, Location loc2, ValueRange ivs,
              ValueRange iters) -> SmallVector<Value> {
            Value batch = ivs[0];
            Value j = ivs[1];

            Value aSlice = b.create<tensor::ExtractSliceOp>(
                loc2, aBatchTy, op.getLeft(), ValueRange{batch, c0, c0},
                ValueRange{}, ValueRange{}, dynOffsets3, staticSizesABatch,
                unitStrides3);
            Value aMat = b.create<tensor::CollapseShapeOp>(loc2, aMatTy, aSlice,
                                                           collapseABatch);

            Value bSlice = b.create<tensor::ExtractSliceOp>(
                loc2, bColTy, op.getRight(), ValueRange{batch, c0, j},
                ValueRange{}, ValueRange{}, dynOffsets3, staticSizesBCol,
                unitStrides3);
            Value x = b.create<tensor::CollapseShapeOp>(loc2, xTy, bSlice,
                                                        collapseBCol);

            Value biasVec;
            if (bias) {
              Value biasSlice = b.create<tensor::ExtractSliceOp>(
                  loc2, biasColTy, bias, ValueRange{batch, c0, j}, ValueRange{},
                  ValueRange{}, dynOffsets3, staticSizesBias, unitStrides3);
              biasVec = b.create<tensor::CollapseShapeOp>(
                  loc2, biasVecTy, biasSlice, collapseBias);
            }

            Value y = b.create<cinm::GemvOp>(loc2, yVecTy, aMat, x,
                                             bias ? biasVec : Value(), Value())
                          .getResult();

            Value yExpanded =
                b.create<tensor::ExpandShapeOp>(loc2, yExpandedTy, y, expandY);

            Value accIn = iters.front();
            Value accOut = b.create<tensor::InsertSliceOp>(
                loc2, yExpanded, accIn, ValueRange{batch, c0, j}, ValueRange{},
                ValueRange{}, dynOffsets3, staticSizesY, unitStrides3);

            return {accOut};
          });
    } else if (splitDim == 1) {
      SmallVector<int64_t, 3> dynOffsets3(3, ShapedType::kDynamic);
      SmallVector<int64_t, 3> unitStrides3{1, 1, 1};

      SmallVector<int64_t, 3> staticSizesARow{1, 1, K};
      auto aRow3DTy = RankedTensorType::get({1, 1, K}, elemTy);
      SmallVector<ReassociationIndices, 2> collapseARow{{0, 1}, {2}};
      auto aRow2DTy = RankedTensorType::get({1, K}, elemTy);

      SmallVector<int64_t, 3> staticSizesBCol{1, K, 1};
      auto bColTy = RankedTensorType::get({1, K, 1}, elemTy);
      SmallVector<ReassociationIndices, 1> collapseBCol{{0, 1, 2}};
      auto xTy = RankedTensorType::get({K}, elemTy);

      SmallVector<int64_t, 3> staticSizesBias{1, 1, 1};
      auto biasScalarTy = RankedTensorType::get({1, 1, 1}, elemTy);
      SmallVector<ReassociationIndices, 1> collapseBias{{0, 1, 2}};
      auto biasVecTy = RankedTensorType::get({1}, elemTy);

      SmallVector<ReassociationIndices, 1> expandScalar{{0, 1, 2}};
      auto yExpandedTy = RankedTensorType::get({1, 1, 1}, elemTy);
      SmallVector<int64_t, 3> staticSizesY{1, 1, 1};
      auto yScalarTy = RankedTensorType::get({1}, elemTy);

      finals = createNestedAffineForLoops(
          rewriter, loc, ArrayRef<int64_t>{B, M, N}, ArrayRef<int64_t>{1, 1, 1},
          ValueRange{acc0},
          [&](OpBuilder &b, Location loc2, ValueRange ivs,
              ValueRange iters) -> SmallVector<Value> {
            Value batch = ivs[0];
            Value i = ivs[1];
            Value j = ivs[2];

            Value aSlice = b.create<tensor::ExtractSliceOp>(
                loc2, aRow3DTy, op.getLeft(), ValueRange{batch, i, c0},
                ValueRange{}, ValueRange{}, dynOffsets3, staticSizesARow,
                unitStrides3);
            Value aRow = b.create<tensor::CollapseShapeOp>(
                loc2, aRow2DTy, aSlice, collapseARow);

            Value bSlice = b.create<tensor::ExtractSliceOp>(
                loc2, bColTy, op.getRight(), ValueRange{batch, c0, j},
                ValueRange{}, ValueRange{}, dynOffsets3, staticSizesBCol,
                unitStrides3);
            Value x = b.create<tensor::CollapseShapeOp>(loc2, xTy, bSlice,
                                                        collapseBCol);

            Value biasVec;
            if (bias) {
              Value biasSlice = b.create<tensor::ExtractSliceOp>(
                  loc2, biasScalarTy, bias, ValueRange{batch, i, j},
                  ValueRange{}, ValueRange{}, dynOffsets3, staticSizesBias,
                  unitStrides3);
              biasVec = b.create<tensor::CollapseShapeOp>(
                  loc2, biasVecTy, biasSlice, collapseBias);
            }

            Value y = b.create<cinm::GemvOp>(loc2, yScalarTy, aRow, x,
                                             bias ? biasVec : Value(), Value())
                          .getResult();

            Value yExpanded = b.create<tensor::ExpandShapeOp>(loc2, yExpandedTy,
                                                              y, expandScalar);

            Value accOut = b.create<tensor::InsertSliceOp>(
                loc2, yExpanded, iters.front(), ValueRange{batch, i, j},
                ValueRange{}, ValueRange{}, dynOffsets3, staticSizesY,
                unitStrides3);

            return {accOut};
          });
    } else {
      return rewriter.notifyMatchFailure(op, "split-dim must be 1 or 2");
    }

    rewriter.replaceOp(op, finals.front());

    if (parentCompute) {
      if (auto ts =
              parentCompute->getAttrOfType<DenseI64ArrayAttr>("tileSizes")) {
        auto arr = ts.asArrayRef();
        if (arr.size() == 4) {
          SmallVector<int64_t, 3> gemvTS;
          if (splitDim == 2)
            gemvTS = {arr[0], arr[1], arr[3]};
          else
            gemvTS = {arr[0], arr[2], arr[3]};

          rewriter.modifyOpInPlace(parentCompute, [&] {
            parentCompute->setAttr("tileSizes", rewriter.getDenseI64ArrayAttr(
                                                    ArrayRef<int64_t>(gemvTS)));
          });
        }
      }
    }

    return success();
  }

  int splitDim;
};

struct BatchGemvToLoopedGemv final : OpRewritePattern<cinm::BatchGemvOp> {
  using OpRewritePattern<cinm::BatchGemvOp>::OpRewritePattern;

  explicit BatchGemvToLoopedGemv(MLIRContext *ctx)
      : OpRewritePattern<cinm::BatchGemvOp>(ctx) {}

  LogicalResult matchAndRewrite(cinm::BatchGemvOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto aTy = dyn_cast<RankedTensorType>(op.getLeft().getType());
    auto xTy = dyn_cast<RankedTensorType>(op.getRight().getType());
    auto yTy = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!aTy || !xTy || !yTy || aTy.getRank() != 3 || xTy.getRank() != 2 ||
        yTy.getRank() != 2)
      return rewriter.notifyMatchFailure(op,
                                         "expected ranked tensors (3D,2D,2D)");

    auto parentCompute = op->getParentOfType<cinm::ComputeOp>();
    if (!parentCompute)
      return rewriter.notifyMatchFailure(op, "requires enclosing cinm.compute");

    const int64_t B = aTy.getDimSize(0);
    const int64_t M = aTy.getDimSize(1);
    const int64_t K = aTy.getDimSize(2);
    const int64_t B2 = xTy.getDimSize(0);
    const int64_t K2 = xTy.getDimSize(1);
    const int64_t B3 = yTy.getDimSize(0);
    const int64_t M2 = yTy.getDimSize(1);
    if (ShapedType::isDynamic(B) || ShapedType::isDynamic(M) ||
        ShapedType::isDynamic(K) || ShapedType::isDynamic(B2) ||
        ShapedType::isDynamic(K2) || ShapedType::isDynamic(B3) ||
        ShapedType::isDynamic(M2))
      return rewriter.notifyMatchFailure(op, "dynamic dims not supported");
    if (B != B2 || B != B3)
      return rewriter.notifyMatchFailure(op, "batch dims must match");
    if (K != K2)
      return rewriter.notifyMatchFailure(op, "inner dims must match (K)");
    if (M != M2)
      return rewriter.notifyMatchFailure(op, "result M dim must match left");

    Type elemTy = aTy.getElementType();
    if (elemTy != xTy.getElementType() || elemTy != yTy.getElementType())
      return rewriter.notifyMatchFailure(op, "element types must match");

    Value bias = op.getBias();
    RankedTensorType biasTy;
    if (bias) {
      biasTy = dyn_cast<RankedTensorType>(bias.getType());
      if (!biasTy || biasTy.getRank() != 2)
        return rewriter.notifyMatchFailure(op, "bias must be ranked 2D tensor");
      if (biasTy.getDimSize(0) != B || biasTy.getDimSize(1) != M)
        return rewriter.notifyMatchFailure(op, "bias dims must match (B,M)");
      if (biasTy.getElementType() != elemTy)
        return rewriter.notifyMatchFailure(op, "bias element type must match");
    }

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    SmallVector<int64_t, 3> dynOffsets3(3, ShapedType::kDynamic);
    SmallVector<int64_t, 3> unitStrides3{1, 1, 1};
    SmallVector<int64_t, 3> staticSizesLeft{1, M, K};
    auto aSliceTy = RankedTensorType::get({1, M, K}, elemTy);
    SmallVector<ReassociationIndices, 2> collapseLeft{{0, 1}, {2}};
    auto matTy = RankedTensorType::get({M, K}, elemTy);

    SmallVector<int64_t, 2> dynOffsets2(2, ShapedType::kDynamic);
    SmallVector<int64_t, 2> unitStrides2{1, 1};
    SmallVector<int64_t, 2> staticSizesVec{1, K};
    auto xSliceTy = RankedTensorType::get({1, K}, elemTy);
    SmallVector<ReassociationIndices, 1> collapseVec{{0, 1}};
    auto vecTy = RankedTensorType::get({K}, elemTy);

    SmallVector<int64_t, 2> staticSizesBias{1, M};
    auto biasSliceTy = RankedTensorType::get({1, M}, elemTy);
    SmallVector<ReassociationIndices, 1> collapseBias{{0, 1}};
    auto biasVecTy = RankedTensorType::get({M}, elemTy);

    SmallVector<ReassociationIndices, 1> expandRes{{0, 1}};
    auto resSliceTy = RankedTensorType::get({1, M}, elemTy);
    SmallVector<int64_t, 2> staticSizesRes{1, M};

    Value acc0 =
        rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{B, M}, elemTy);

    SmallVector<Value> finals = createNestedAffineForLoops(
        rewriter, loc, ArrayRef<int64_t>{B}, ArrayRef<int64_t>{1},
        ValueRange{acc0},
        [&](OpBuilder &b, Location loc2, ValueRange ivs,
            ValueRange iters) -> SmallVector<Value> {
          Value batch = ivs.front();

          Value aSlice = b.create<tensor::ExtractSliceOp>(
              loc2, aSliceTy, op.getLeft(), ValueRange{batch, c0, c0},
              ValueRange{}, ValueRange{}, dynOffsets3, staticSizesLeft,
              unitStrides3);
          Value aMat = b.create<tensor::CollapseShapeOp>(loc2, matTy, aSlice,
                                                         collapseLeft);

          Value xSlice = b.create<tensor::ExtractSliceOp>(
              loc2, xSliceTy, op.getRight(), ValueRange{batch, c0},
              ValueRange{}, ValueRange{}, dynOffsets2, staticSizesVec,
              unitStrides2);
          Value xVec = b.create<tensor::CollapseShapeOp>(loc2, vecTy, xSlice,
                                                         collapseVec);

          Value biasVec;
          if (bias) {
            Value biasSlice = b.create<tensor::ExtractSliceOp>(
                loc2, biasSliceTy, bias, ValueRange{batch, c0}, ValueRange{},
                ValueRange{}, dynOffsets2, staticSizesBias, unitStrides2);
            biasVec = b.create<tensor::CollapseShapeOp>(
                loc2, biasVecTy, biasSlice, collapseBias);
          }

          auto yTy = RankedTensorType::get({M}, elemTy);
          Value y = b.create<cinm::GemvOp>(loc2, yTy, aMat, xVec,
                                           bias ? biasVec : Value(), Value())
                        .getResult();

          Value yExpanded =
              b.create<tensor::ExpandShapeOp>(loc2, resSliceTy, y, expandRes);

          Value accOut = b.create<tensor::InsertSliceOp>(
              loc2, yExpanded, iters.front(), ValueRange{batch, c0},
              ValueRange{}, ValueRange{}, dynOffsets2, staticSizesRes,
              unitStrides2);
          return {accOut};
        });

    rewriter.replaceOp(op, finals.front());

    if (parentCompute) {
      if (auto ts =
              parentCompute->getAttrOfType<DenseI64ArrayAttr>("tileSizes")) {
        auto arr = ts.asArrayRef();
        if (arr.size() == 3) {
          SmallVector<int64_t, 2> gemvTS{arr[1], arr[2]};
          rewriter.modifyOpInPlace(parentCompute, [&] {
            parentCompute->setAttr("tileSizes", rewriter.getDenseI64ArrayAttr(
                                                    ArrayRef<int64_t>(gemvTS)));
          });
        }
      }
    }

    return success();
  }
};

struct CinmGemmToLoopedGemvPass
    : public impl::CinmGemmToLoopedGemvPassBase<CinmGemmToLoopedGemvPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    tensor::TensorDialect, cinm::CinmDialect>();
  }

  void runOnOperation() final {
    if (splitDimOpt != 1 && splitDimOpt != 2) {
      getOperation()->emitError()
          << "cinm-gemm-to-gemv: --split-dim must be 1 (i/M) or 2 (j/N); "
             "reduction dim (k) is not allowed";
      signalPassFailure();
      return;
    }

    RewritePatternSet ps(&getContext());
    ps.add<GemmToLoopedGemv>(&getContext(), splitDimOpt);
    ps.add<BatchGemmToLoopedGemv>(&getContext(), splitDimOpt);
    ps.add<BatchGemvToLoopedGemv>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(ps))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::cinm
