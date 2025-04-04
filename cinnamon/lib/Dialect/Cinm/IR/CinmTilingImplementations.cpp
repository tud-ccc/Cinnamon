#include "cinm-mlir/Conversion/CommonPatterns.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/Interfaces/TilingInterface.h"

#include <array>
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/AffineExpr.h>
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
                                                getDimensionsAttr(), reduceClusterSize)});
  } else if (method == ReduceMethod::MUL) {
    return TilingResult2({createVectorReduceMul(builder, getLoc(), getOperand(),
                                                getDimensionsAttr(), reduceClusterSize)});
  } else if (method == ReduceMethod::MAX) {
    return TilingResult2({createVectorReduceMax(builder, getLoc(), getOperand(),
                                                getDimensionsAttr(), reduceClusterSize)});
  } else if (method == ReduceMethod::MIN) {
    return TilingResult2({createVectorReduceMin(builder, getLoc(), getOperand(),
                                                getDimensionsAttr(), reduceClusterSize)});
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
      cast<RankedTensorType>(getResult().getType());
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
        DenseElementsAttr zeros;
        if (auto floatType =
                dyn_cast<FloatType>(reductionAccTy.getElementType())) {
          zeros = DenseElementsAttr::get(
              reductionAccTy,
              {APFloat::getZero(floatType.getFloatSemantics())});
        } else {
          zeros = DenseElementsAttr::get(
              reductionAccTy,
              {APInt::getZero(reductionAccTy.getElementTypeBitWidth())});
        }

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

TilingResult2 Elementwise_Unary_Op::convertToTiledOps(OpBuilder &builder0, TilingParameters params) {
  ImplicitLocOpBuilder builder(getLoc(), builder0);
  Value input = getInput();

  auto tensorTy = cast<RankedTensorType>(input.getType());
  const auto wgSize = params.workingGroupSize();
  const auto numElements = tensorTy.getNumElements();
  if (wgSize > numElements) {
    return emitWarning("Cannot tile, working group is too large");
  }

  // This is the max number of reductions we can theoretically do on
  // a single CNM.launch.
  auto reduceClusterSize =
      params.reduceClusterSize(2, numElements, tensorTy.getElementType());

  // We need the actual tile size to not exceed that number, and
  // be able to divide the input by the working group size.
  if (reduceClusterSize * wgSize >= numElements) {
    // we have too much compute
    const auto numLeavesNeeded = numElements / reduceClusterSize;
    emitRemark("This computation could be done with a smaller working group, theoretically as few as " +
        std::to_string(numLeavesNeeded) + " leaves (we have " + std::to_string(wgSize) + ").");
    reduceClusterSize = numElements / wgSize;
  }
  if (numElements % wgSize != 0) {
    return emitWarning("Cannot tile, working group does not divide tensor size");
  }

  auto shape = tensorTy.getShape();
  const RankedTensorType originalType = tensorTy;
  Value originalShapeValue;
  if (shape.size() > 1) {
    // then flatten the tensors first
    originalShapeValue = builder.create<arith::ConstantOp>(
        RankedTensorType::get({static_cast<int64_t>(shape.size())}, builder.getI64Type()),
        builder.getI64TensorAttr(shape));
    input = reshapeStatic(builder, builder.getLoc(), getInput(), {tensorTy.getNumElements()});

    tensorTy = cast<RankedTensorType>(input.getType());
    shape = tensorTy.getShape();
  }

  const Value resultInit = builder.create<tensor::EmptyOp>(tensorTy, ValueRange{});
  const auto tileSize = reduceClusterSize * wgSize;

  SmallVector<Value> result = createNestedAffineForLoops(
      builder, getLoc(), {tensorTy.getNumElements()}, {tileSize},
      ValueRange{resultInit},
      [&](OpBuilder &builder, Location loc, ValueRange indices, ValueRange iterArgs) -> SmallVector<Value> {
        const SmallVector<int64_t, 2> offsets{ShapedType::kDynamic};
        const SmallVector<int64_t, 2> sizes{tileSize};
        const SmallVector<int64_t, 2> strides{1};

        const auto sliceTy = RankedTensorType::get({tileSize}, tensorTy.getElementType());

        const Value inputSlice = builder.create<tensor::ExtractSliceOp>(
            loc, sliceTy, input, indices, ValueRange{}, ValueRange{}, offsets,
            sizes, strides);

        // here we recreate the same op with smaller dimensions
        auto smaller = builder.create<Elementwise_Unary_Op>(loc, getMethod(), inputSlice);
        markOpAsNoTile(smaller);

        const Value subResult = builder.create<tensor::InsertSliceOp>(
            loc, smaller.getResult(), iterArgs[0], indices, ValueRange{},
            ValueRange{}, offsets, sizes, strides);

        return {subResult};
      });
  if (originalType.getShape().size() > 1) {
    result[0] = builder.create<tensor::ReshapeOp>(originalType, result[0],
                                                  originalShapeValue);
  }
  return TilingResult2(result);
}

template <typename OP, bool IsScalarOp>
TilingResult2 tileElementWiseBinaryOp(OpBuilder &builder0, OP op,
                                      TilingParameters params) {
  ImplicitLocOpBuilder builder(op.getLoc(), builder0);
  Value lhs = op.getOperand(0);
  Value rhs = op.getOperand(1);

  RankedTensorType tensorTy = cast<RankedTensorType>(lhs.getType());
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

    if constexpr (!IsScalarOp) {
      rhs = cinm::reshapeStatic(builder, builder.getLoc(), op.getRhs(),
                                {tensorTy.getNumElements()});
    }
    tensorTy = cast<RankedTensorType>(lhs.getType());
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

        const RankedTensorType sliceTy =
            RankedTensorType::get({tileSize}, tensorTy.getElementType());

        const Value lhsSlice = builder.create<tensor::ExtractSliceOp>(
            loc, sliceTy, lhs, indices, ValueRange{}, ValueRange{}, offsets,
            sizes, strides);

        Value rhsSlice = rhs;
        if constexpr (!IsScalarOp) {
          rhsSlice = builder.create<tensor::ExtractSliceOp>(
              loc, sliceTy, rhs, indices, ValueRange{}, ValueRange{}, offsets,
              sizes, strides);
        }

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
  return tileElementWiseBinaryOp<AddOp, false>(builder, *this, params);
}

TilingResult2 AddsOp::convertToTiledOps(OpBuilder &builder,
                                        TilingParameters params) {
  return tileElementWiseBinaryOp<AddsOp, true>(builder, *this, params);
}

TilingResult2 SubOp::convertToTiledOps(OpBuilder &builder,
                                       TilingParameters params) {
  return tileElementWiseBinaryOp<SubOp, false>(builder, *this, params);
}

TilingResult2 SubsOp::convertToTiledOps(OpBuilder &builder,
                                        TilingParameters params) {
  return tileElementWiseBinaryOp<SubsOp, true>(builder, *this, params);
}

TilingResult2 MulOp::convertToTiledOps(OpBuilder &builder,
                                       TilingParameters params) {
  return tileElementWiseBinaryOp<MulOp, false>(builder, *this, params);
}

TilingResult2 MulsOp::convertToTiledOps(OpBuilder &builder,
                                        TilingParameters params) {
  return tileElementWiseBinaryOp<MulsOp, true>(builder, *this, params);
}

TilingResult2 DivOp::convertToTiledOps(OpBuilder &builder,
                                       TilingParameters params) {
  return tileElementWiseBinaryOp<DivOp, false>(builder, *this, params);
}

TilingResult2 DivsOp::convertToTiledOps(OpBuilder &builder,
                                        TilingParameters params) {
  return tileElementWiseBinaryOp<DivsOp, true>(builder, *this, params);
}
