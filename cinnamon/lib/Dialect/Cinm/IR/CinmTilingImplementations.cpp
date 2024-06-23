

#include "cinm-mlir/Conversion/CommonPatterns.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/Interfaces/TilingInterface.h"

#include <array>
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/IR/ValueRange.h>

using namespace mlir;
using namespace mlir::cinm;

// TilingResult is already a thing in MLIR
using TilingResult2 = FailureOr<SmallVector<Value>>;

TilingResult2 ReduceOp::convertToTiledOps(OpBuilder &builder,
                                          TilingParameters params) {
  auto ty = getInput().getType();
  auto reduceClusterSize =
      params.reduceClusterSize(1, ty.getNumElements(), ty.getElementType());

  auto method = getMethod();
  if (method == ReduceMethod::ADD) {
    return TilingResult2({createVectorReduceAdd(builder, getLoc(), getOperand(),
                                                reduceClusterSize)});
  } else if (method == ReduceMethod::MUL) {
    return TilingResult2({createVectorReduceMul(builder, getLoc(), getOperand(),
                                                reduceClusterSize)});
  } else if (method == ReduceMethod::MAX) {
    return TilingResult2({createVectorReduceMax(builder, getLoc(), getOperand(),
                                                reduceClusterSize)});
  } else if (method == ReduceMethod::MIN) {
    return TilingResult2({createVectorReduceMin(builder, getLoc(), getOperand(),
                                                reduceClusterSize)});
  } else {
    abort();
  }
}

static constexpr std::array<int64_t, 2> noStaticOffsets{ShapedType::kDynamic,
                                                        ShapedType::kDynamic};

static constexpr std::array<int64_t, 2> unitStrides{1, 1};

TilingResult2 GemmOp::convertToTiledOps(OpBuilder &builder,
                                        TilingParameters params) {
  Value lhs = getLeft();
  Value rhs = getRight();

  const RankedTensorType lhsType = getLeft().getType();
  const RankedTensorType rhsType = getRight().getType();

  const RankedTensorType resultType =
      getResult().getType().cast<RankedTensorType>();
  const ArrayRef<int64_t> resultShape = resultType.getShape();

  const Value resultInit = builder.create<tensor::EmptyOp>(
      getLoc(), resultShape, resultType.getElementType());

  auto rDim = lhsType.getDimSize(1);
  auto eltTy = rhsType.getElementType();

  // Size of the tile on the reduction dimension.
  auto r = params.reduceClusterSize(2, rDim, lhsType.getElementType(),
                                    /*extraElements=*/1);
  auto parallelTileSizes =
      params.parallelClusterSize(lhsType.getDimSize(0), rhsType.getDimSize(1));
  if (!parallelTileSizes)
    return failure();

  // Tile sizes of each parallel dimension
  const auto [p0, p1] = *parallelTileSizes;

  auto res = createNestedAffineForLoops(
      builder, getLoc(), resultShape, {p0, p1}, ValueRange{resultInit},
      [&, p0, p1](OpBuilder &builder, Location loc, ValueRange indices,
                  ValueRange iterArgs) -> SmallVector<Value> {
        const auto parIndices = indices;
        const SmallVector<int64_t, 2> resultSizes{p0, p1};
        const ValueRange resultDynamicOffsets = parIndices;

        auto reductionAccTy = RankedTensorType::get({p0, p1}, eltTy);
        auto zeros = DenseIntElementsAttr::get(reductionAccTy, {0});
        Value cst0 = builder.create<arith::ConstantOp>(loc, zeros);

        // this is the reduction loop
        SmallVector<Value, 1> reductionResult = createNestedAffineForLoops(
            builder, loc, {rDim}, {r}, cst0,
            [&, p0, p1](OpBuilder &builder, Location loc, ValueRange indices,
                        ValueRange iterArgs) -> SmallVector<Value> {
              const auto indexInRedDim = indices[0];

              const SmallVector<int64_t, 2> lhsSizes{p0, r};

              const Type lhsSliceType = RankedTensorType::get({p0, r}, eltTy);

              const Value lhsSlice = builder.create<tensor::ExtractSliceOp>(
                  loc, lhsSliceType, lhs,
                  ValueRange{parIndices[0], indexInRedDim}, ValueRange{},
                  ValueRange{}, ArrayRef(noStaticOffsets), lhsSizes,
                  ArrayRef(unitStrides));

              const Type rhsSliceType = RankedTensorType::get({r, p1}, eltTy);

              const SmallVector<int64_t, 2> rhsSizes{r, p1};

              const Value rhsSlice = builder.create<tensor::ExtractSliceOp>(
                  loc, rhsSliceType, rhs,
                  ValueRange{indexInRedDim, parIndices[1]}, ValueRange{},
                  ValueRange{}, ArrayRef(noStaticOffsets), rhsSizes,
                  ArrayRef(unitStrides));

              // now we have a LHS tile <reduceClusterSize>
              // and RHS tile <reduceClusterSize x parallelTileSize

              // Here we're back to doing
              // GEMM(ltile: <p0 x r>, rtile: <r x p1>) + iterArgs[0]
              auto tmpReduce = builder.create<cinm::GemmOp>(
                  loc, lhsSlice, rhsSlice, iterArgs[0]);
              cinm::markOpAsNoTile(tmpReduce);
              return {tmpReduce};
            });

        const Value result = builder.create<tensor::InsertSliceOp>(
            loc, reductionResult[0], iterArgs[0], resultDynamicOffsets,
            ValueRange{}, ValueRange{}, ArrayRef(noStaticOffsets), resultSizes,
            ArrayRef(unitStrides));
        return {result};
      });
  return TilingResult2(res);
}

// convert to a gemm, gemm pattern will be applied after
TilingResult2 GemvOp::convertToTiledOps(OpBuilder &builder, TilingParameters) {
  auto rhsAsMatrix = cinm::reshapeStatic(
      builder, getLoc(), getRight(), {getRight().getType().getDimSize(0), 1});
  auto gemm = builder.create<cinm::GemmOp>(getLoc(), getLeft(), rhsAsMatrix);
  Value toVector = cinm::reshapeStatic(builder, getLoc(), gemm.getResult(),
                                       getResult().getType().getShape());
  return FailureOr<SmallVector<Value>>({toVector});
}

template <typename OP>
TilingResult2 tileElementWiseBinaryOp(OpBuilder &builder0, OP op,
                                      TilingParameters params) {
  ImplicitLocOpBuilder builder(op.getLoc(), builder0);
  Value lhs = op.getOperand(0);
  Value rhs = op.getOperand(1);

  RankedTensorType tensorTy = lhs.getType().cast<RankedTensorType>();
  auto wgSize = params.workingGroupSize();
  auto numElements = tensorTy.getNumElements();
  if (wgSize > numElements) {
    return emitWarning(op.getLoc())
           << "Cannot tile, working group is too large";
  }

  // This is the max number of reductions we can theoretically do on
  // a single CNM.launch.
  auto reduceClusterSize =
      params.reduceClusterSize(3, numElements, tensorTy.getElementType());
  // We need the actual tile size to not exceed that number, and
  // be able to divide the input by the working group size.
  if (reduceClusterSize * wgSize >= numElements) {
    // we have too much compute
    auto numLeavesNeeded = numElements / reduceClusterSize;
    emitRemark(op.getLoc())
        << "This computation could be done with a smaller working group,"
           " theoretically as few as "
        << numLeavesNeeded << " leaves (we have " << wgSize << ").";
    reduceClusterSize = numElements / wgSize;
  }
  if (numElements % wgSize != 0)
    return emitWarning(op.getLoc())
           << "Cannot tile, working group does not divide tensor size";

  auto shape = tensorTy.getShape();
  const RankedTensorType originalType = tensorTy;
  Value originalShapeValue;
  if (shape.size() > 1) {
    // then flatten the tensors first
    originalShapeValue = builder.create<arith::ConstantOp>(
        RankedTensorType::get({static_cast<int64_t>(shape.size())},
                              builder.getI64Type()),
        builder.getI64TensorAttr(shape));
    lhs = cinm::reshapeStatic(builder, builder.getLoc(), op.getLhs(),
                              {tensorTy.getNumElements()});
    rhs = cinm::reshapeStatic(builder, builder.getLoc(), op.getRhs(),
                              {tensorTy.getNumElements()});
    tensorTy = lhs.getType().cast<RankedTensorType>();
    shape = tensorTy.getShape();
  }

  const Value resultInit =
      builder.create<tensor::EmptyOp>(tensorTy, ValueRange{});
  const auto tileSize = reduceClusterSize * wgSize;

  SmallVector<Value> result = createNestedAffineForLoops(
      builder, op.getLoc(), {tensorTy.getNumElements()}, {tileSize},
      ValueRange{resultInit},
      [&](OpBuilder &builder, Location loc, ValueRange indices,
          ValueRange iterArgs) -> SmallVector<Value> {
        const SmallVector<int64_t, 2> offsets{ShapedType::kDynamic};
        const SmallVector<int64_t, 2> sizes{tileSize};
        const SmallVector<int64_t, 2> strides{1};

        const RankedTensorType sliceTy = RankedTensorType::get(
            {tileSize}, tensorTy.getElementType());

        const Value lhsSlice = builder.create<tensor::ExtractSliceOp>(
            loc, sliceTy, lhs, indices, ValueRange{}, ValueRange{}, offsets,
            sizes, strides);
        const Value rhsSlice = builder.create<tensor::ExtractSliceOp>(
            loc, sliceTy, rhs, indices, ValueRange{}, ValueRange{}, offsets,
            sizes, strides);

        // here we recreate the same op with smaller dimensions
        OP smaller = builder.create<OP>(loc, lhsSlice, rhsSlice);
        markOpAsNoTile(smaller);

        const Value result = builder.create<tensor::InsertSliceOp>(
            loc, smaller.getResult(), iterArgs[0], indices, ValueRange{},
            ValueRange{}, offsets, sizes, strides);

        return {result};
      });
  if (originalType.getShape().size() > 1) {
    result[0] = builder.create<tensor::ReshapeOp>(originalType, result[0],
                                                  originalShapeValue);
  }
  return TilingResult2(result);
}

TilingResult2 AddOp::convertToTiledOps(OpBuilder &builder,
                                       TilingParameters params) {
  return tileElementWiseBinaryOp<AddOp>(builder, *this, params);
}

TilingResult2 SubOp::convertToTiledOps(OpBuilder &builder,
                                       TilingParameters params) {
  return tileElementWiseBinaryOp<SubOp>(builder, *this, params);
}

TilingResult2 MulOp::convertToTiledOps(OpBuilder &builder,
                                       TilingParameters params) {
  return tileElementWiseBinaryOp<MulOp>(builder, *this, params);
}

TilingResult2 DivOp::convertToTiledOps(OpBuilder &builder,
                                       TilingParameters params) {
  return tileElementWiseBinaryOp<DivOp>(builder, *this, params);
}
