

#include "cinm-mlir/Conversion/CommonPatterns.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/Interfaces/TilingInterface.h"

#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>

using namespace mlir;
using namespace mlir::cinm;

SmallVector<Value> MinOp::convertToTiledOps(OpBuilder &builder,
                                            TilingParameters params) {
  auto ty = getInput().getType();
  return {createVectorReduceMin(
      builder, getLoc(), getOperand(),
      params.reduceClusterSize(1, ty.getNumElements(), ty.getElementType()))};
}

SmallVector<Value> MaxOp::convertToTiledOps(OpBuilder &builder,
                                            TilingParameters params) {
  auto ty = getInput().getType();
  return {createVectorReduceMax(
      builder, getLoc(), getOperand(),
      params.reduceClusterSize(1, ty.getNumElements(), ty.getElementType()))};
}

SmallVector<Value> ReduceOp::convertToTiledOps(OpBuilder &builder,
                                               TilingParameters params) {
  auto ty = getInput().getType();
  auto reduceClusterSize =
      params.reduceClusterSize(1, ty.getNumElements(), ty.getElementType());
  if (getMethod() == ReduceMethod::ADD) {
    return {createVectorReduceAdd(builder, getLoc(), getOperand(),
                                  reduceClusterSize)};
  } else if (getMethod() == ReduceMethod::MUL) {
    return {createVectorReduceMul(builder, getLoc(), getOperand(),
                                  reduceClusterSize)};
  } else {
    abort();
  }
}

Value scatterTensorIntoMemref(OpBuilder &builder, Value value,
                              int64_t numCopies) {
  auto inputTy = value.getType().cast<RankedTensorType>();
  assert(inputTy.getRank() == 2 && inputTy.getDimSize(0) == 1);
  // layout map
  // (d0, d1) -> (d1)
  auto layoutMap =
      AffineMap::get(2, 0, getAffineDimExpr(1, value.getContext()));

  auto baseMemrefTy =
      MemRefType::get(inputTy.getShape(), inputTy.getElementType());
  auto viewTy = MemRefType::get({numCopies, inputTy.getDimSize(1)},
                                inputTy.getElementType(), layoutMap);

  auto baseMemref =
      builder.create<memref::AllocOp>(value.getLoc(), baseMemrefTy);

  auto scatteredMemref = builder.create<memref::SubViewOp>(
      value.getLoc(), viewTy, baseMemref, ArrayRef<int64_t>{0, 0},
      ArrayRef<int64_t>{numCopies, inputTy.getDimSize(1)},
      ArrayRef<int64_t>{1, 1});

  return scatteredMemref;
}

/** This tiling works this way:
    - Given a Gemm of dimensions (%A: <MxR>), (%B: <RxN>) -> <MxN>
    - If M == 1 then
      - determine optimal tile sizes RT and NT for the R and N dimensions
      - then emit the following program:
        affine.loop %j = 0 to N step NT iter_args(%acc0 = empty: <1xN>){
          %tile: <1xNT> = affine.loop %k = 0 to R step RT iter_args(%acc =
   zeros: <1xNT>) { %rowTile = slice %A[0, %k] [1, RT] [1, 1] : <1xRT> %colTile
   = slice %B[%k, %j] [RT, NT] [1, 1] : <RTxNT> %tmp = cinm.gemm %rowTile,
   %colTile : -> <1xNT> %add = cinm.add %acc, tmp affine.yield %add : <1xNT>
          }
          %tmp = insert_slice %tile, %acc0[0, %j] [1, NT] [1, 1]
          affine.yield %tmp
        }
    - if M > 1 then the gemm is first reduced into a loop over M, and gemms of
   size <1xR> <RxN>

 */
SmallVector<Value> GemmOp::convertToTiledOps(OpBuilder &builder,
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

  // Size of the tile on the reduction dimension.
  auto reduceClusterSize = params.reduceClusterSize(
      2, resultType.getNumElements(), lhsType.getElementType());
  auto parallelTileSize =
      params.parallelClusterSize(rhsType.getDimSize(1), reduceClusterSize);

  if (lhsType.getDimSize(0) == 1) {
    // then it's basically vector-mat multiplication
    // arguments have shape <1xR>, <RxB>
    auto r = lhsType.getDimSize(1);
    auto b = rhsType.getDimSize(1);
    auto eltTy = rhsType.getElementType();

    // iterate over b (parallel loop)
    return createNestedAffineForLoops(
        builder, getLoc(), {b}, {parallelTileSize}, {resultInit},
        [&](OpBuilder &builder, Location loc, ValueRange indices,
            ValueRange iterArgs) -> SmallVector<Value> {
          auto reductionAccTy =
              RankedTensorType::get({1, parallelTileSize}, eltTy);
          auto cst0 = builder.create<arith::ConstantOp>(
              loc, builder.getZeroAttr(eltTy));
          auto reductionAccInit =
              builder.create<tensor::SplatOp>(loc, cst0, reductionAccTy);

          const auto indexInParDim = indices[0];

          // this is the reduction loop
          SmallVector<Value, 1> reductionResult = createNestedAffineForLoops(
              builder, loc, {r}, {reduceClusterSize},
              reductionAccInit->getResults(),
              [&](OpBuilder &builder, Location loc, ValueRange indices,
                  ValueRange iterArgs) -> SmallVector<Value> {
                const auto indexInRedDim = indices[0];

                const ArrayRef<int64_t> lhsOffsets{0, ShapedType::kDynamic};
                const ArrayRef<int64_t> lhsSizes{1, reduceClusterSize};
                const ArrayRef<int64_t> lhsStrides{1, 1};

                const Type lhsSliceType =
                    RankedTensorType::get({1, reduceClusterSize}, eltTy);

                const ArrayRef<Value> lhsDynamicOffsets{indexInRedDim};
                const Value lhsSlice = builder.create<tensor::ExtractSliceOp>(
                    loc, lhsSliceType, lhs, lhsDynamicOffsets, ValueRange{},
                    ValueRange{}, lhsOffsets, lhsSizes, lhsStrides);

                // todo this is still a square tile but the left tile is flat
                const Type rhsSliceType = RankedTensorType::get(
                    {reduceClusterSize, parallelTileSize}, eltTy);

                const ArrayRef<int64_t> rhsOffsets{ShapedType::kDynamic,
                                                   ShapedType::kDynamic};
                const ArrayRef<int64_t> rhsSizes{reduceClusterSize,
                                                 parallelTileSize};
                const ArrayRef<int64_t> rhsStrides{1, 1};

                const Value rhsSlice = builder.create<tensor::ExtractSliceOp>(
                    loc, rhsSliceType, rhs,
                    ValueRange{indexInParDim, indexInRedDim}, ValueRange{},
                    ValueRange{}, rhsOffsets, rhsSizes, rhsStrides);

                // now we have a LHS tile <reduceClusterSize>
                // and RHS tile <reduceClusterSize x parallelTileSize

                // Here we're back to doing
                // GEMM(ltile: <1 x rcs>, rtile: <rcs x pts>) + iterArgs[0]
                //                auto tmpReduce = builder.create<cinm::GemmOp>(
                // loc, lhsSlice, rhsSlice, iterArgs[0]);
                //              cinm::markOpAsNoTile(tmpReduce);

                const auto rowsEmpty = builder.create<memref::AllocOp>(
                    loc, MemRefType::get(rhsSlice.getType()));
                const auto rowsFull = builder.create<tensor::ScatterOp>(loc);
                const

                    linalg::ReduceOp reduceOp =
                        builder.create<linalg::ReduceOp>(
                            loc, ValueRange{reshape0, reshape1},
                            ValueRange{outTensor}, ArrayRef<int64_t>{1},
                            [&](OpBuilder &builder, Location loc,
                                ValueRange args) {
                              const Value tmp =
                                  merge2(builder, loc, args[0], args[1]);
                              const Value result =
                                  reduce(builder, loc, tmp, args[2]);
                              builder.create<linalg::YieldOp>(loc, result);
                            });

                return {tmpReduce};
              });

          const ArrayRef<int64_t> resultOffsets{0, ShapedType::kDynamic};
          const ArrayRef<int64_t> &resultSizes{1, parallelTileSize};
          const ArrayRef<int64_t> resultStrides{1, 1};
          const ValueRange resultDynamicOffsets{indexInParDim};

          const Value result = builder.create<tensor::InsertSliceOp>(
              loc, reductionResult[0], iterArgs[0], resultDynamicOffsets,
              ValueRange{}, ValueRange{}, resultOffsets, resultSizes,
              resultStrides);

          return {result};
        });
  }

  const ArrayRef<int64_t> tileSizes = {1, parallelTileSize};

  return createNestedAffineForLoops(
      builder, getLoc(), resultShape, tileSizes, ValueRange{resultInit},
      [&](OpBuilder &builder, Location loc, ValueRange indices,
          ValueRange iterArgs) -> SmallVector<Value> {
        const ArrayRef<int64_t> lhsOffsets{ShapedType::kDynamic, 0};
        const ArrayRef<int64_t> lhsSizes{tileSizes[0], lhsType.getDimSize(1)};
        const ArrayRef<int64_t> unitStrides{1, 1};
        const ArrayRef<int64_t> &lhsStrides = unitStrides;
        const ArrayRef<int64_t> rhsOffsets{0, ShapedType::kDynamic};
        const ArrayRef<int64_t> rhsSizes{rhsType.getDimSize(0), tileSizes[1]};
        const ArrayRef<int64_t> &rhsStrides = unitStrides;
        const ArrayRef<int64_t> resultOffsets{ShapedType::kDynamic,
                                              ShapedType::kDynamic};
        const ArrayRef<int64_t> &resultSizes = tileSizes;
        const ArrayRef<int64_t> resultStrides = unitStrides;

        const SmallVector<Value> lhsDynamicOffsets{indices[0]};
        const RankedTensorType lhsSliceType =
            tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
                2, lhsType, lhsOffsets, lhsSizes, lhsStrides);
        const Value lhsSlice = builder.create<tensor::ExtractSliceOp>(
            loc, lhsSliceType, lhs, lhsDynamicOffsets, ValueRange{},
            ValueRange{}, lhsOffsets, lhsSizes, lhsStrides);

        const SmallVector<Value> rhsDynamicOffsets{indices[1]};
        const Type rhsSliceType =
            tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
                2, rhsType, rhsOffsets, rhsSizes, rhsStrides);
        const Value rhsSlice = builder.create<tensor::ExtractSliceOp>(
            loc, rhsSliceType, rhs, rhsDynamicOffsets, ValueRange{},
            ValueRange{}, rhsOffsets, rhsSizes, rhsStrides);

        // will be tiled further
        GemmOp smallerGemm =
            builder.create<cinm::GemmOp>(loc, lhsSlice, rhsSlice);

        const Value result = builder.create<tensor::InsertSliceOp>(
            loc, smallerGemm.getResult(), iterArgs[0], indices, ValueRange{},
            ValueRange{}, resultOffsets, resultSizes, resultStrides);

        return {result};
      });
}

// convert to a gemm, gemm pattern will be applied after
SmallVector<Value> GemvOp::convertToTiledOps(OpBuilder &builder,
                                             TilingParameters) {
  auto rhsAsMatrix = cinm::reshapeStatic(
      builder, getLoc(), getRight(), {getRight().getType().getDimSize(0), 1});
  auto gemm = builder.create<cinm::GemmOp>(getLoc(), getLeft(), rhsAsMatrix);
  Value toVector = cinm::reshapeStatic(builder, getLoc(), gemm.getResult(),
                                       getRight().getType().getShape());
  return {toVector};
}

template <typename OP>
SmallVector<Value> tileElementWiseBinaryOp(OpBuilder &builder0, OP op,
                                           TilingParameters params) {
  ImplicitLocOpBuilder builder(op.getLoc(), builder0);
  Value lhs = op.getOperand(0);
  Value rhs = op.getOperand(1);

  RankedTensorType tensorTy = lhs.getType().cast<RankedTensorType>();
  if (params.workingGroupSize() > tensorTy.getNumElements()) {
    // this is dead code as the op is marked dynamically legal in this case
    assert(false && "Working group is too large");
  }

  auto reduceClusterSize = params.reduceClusterSize(
      3, tensorTy.getNumElements(), tensorTy.getElementType());
  reduceClusterSize *= params.workingGroupSize();
  assert(reduceClusterSize <= tensorTy.getNumElements());

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

  const SmallVector<int64_t, 2> tileSizes = {
      tensorTy.getNumElements() / reduceClusterSize, reduceClusterSize};

  SmallVector<Value, 1> result = createNestedAffineForLoops(
      builder, op.getLoc(), {tensorTy.getNumElements()}, {reduceClusterSize},
      ValueRange{resultInit},
      [&](OpBuilder &builder, Location loc, ValueRange indices,
          ValueRange iterArgs) -> SmallVector<Value> {
        const SmallVector<int64_t, 2> offsets{ShapedType::kDynamic};
        const SmallVector<int64_t, 2> sizes{reduceClusterSize};
        const SmallVector<int64_t, 2> strides{1};

        const RankedTensorType sliceTy = RankedTensorType::get(
            {reduceClusterSize}, tensorTy.getElementType());

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
  return result;
}

SmallVector<Value> AddOp::convertToTiledOps(OpBuilder &builder,
                                            TilingParameters params) {
  return tileElementWiseBinaryOp<AddOp>(builder, *this, params);
}

SmallVector<Value> SubOp::convertToTiledOps(OpBuilder &builder,
                                            TilingParameters params) {
  return tileElementWiseBinaryOp<SubOp>(builder, *this, params);
}

SmallVector<Value> MulOp::convertToTiledOps(OpBuilder &builder,
                                            TilingParameters params) {
  return tileElementWiseBinaryOp<MulOp>(builder, *this, params);
}

SmallVector<Value> DivOp::convertToTiledOps(OpBuilder &builder,
                                            TilingParameters params) {
  return tileElementWiseBinaryOp<DivOp>(builder, *this, params);
}
