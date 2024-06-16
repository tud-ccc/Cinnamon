

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
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/IR/ValueRange.h>

using namespace mlir;
using namespace mlir::cinm;

SmallVector<Value> ReduceOp::convertToTiledOps(OpBuilder &builder,
                                               TilingParameters params) {
  auto ty = getInput().getType();
  auto reduceClusterSize =
      params.reduceClusterSize(1, ty.getNumElements(), ty.getElementType());

  auto method = getMethod();
  if (method == ReduceMethod::ADD) {
    return {createVectorReduceAdd(builder, getLoc(), getOperand(),
                                  reduceClusterSize)};
  } else if (method == ReduceMethod::MUL) {
    return {createVectorReduceMul(builder, getLoc(), getOperand(),
                                  reduceClusterSize)};
  } else if (method == ReduceMethod::MAX) {
    return {createVectorReduceMax(builder, getLoc(), getOperand(),
                                  reduceClusterSize)};
  } else if (method == ReduceMethod::MIN) {
    return {createVectorReduceMin(builder, getLoc(), getOperand(),
                                  reduceClusterSize)};
  } else {
    abort();
  }
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

  auto rDim = lhsType.getDimSize(1);
  auto eltTy = rhsType.getElementType();

  // Size of the tile on the reduction dimension.
  auto r = params.reduceClusterSize(2, rDim, lhsType.getElementType());
  // Tile sizes of each parallel dimension
  auto [p0, p1] =
      params.parallelClusterSize(lhsType.getDimSize(0), rhsType.getDimSize(1));

  return createNestedAffineForLoops(
      builder, getLoc(), resultShape, {p0, p1}, ValueRange{resultInit},
      [&, p0, p1](OpBuilder &builder, Location loc, ValueRange indices,
          ValueRange iterArgs) -> SmallVector<Value> {
        const auto parIndices = indices;
        const ArrayRef<int64_t> unitStrides{1, 1};
        const ArrayRef<int64_t> noStaticOffsets{ShapedType::kDynamic,
                                                ShapedType::kDynamic};

        const ArrayRef<int64_t> resultOffsets = noStaticOffsets;
        const ArrayRef<int64_t> resultSizes = {p0, p1};
        const ValueRange resultDynamicOffsets = parIndices;

        auto reductionAccTy = RankedTensorType::get({p0, p1}, eltTy);
        auto cst0 =
            builder.create<arith::ConstantOp>(loc, builder.getZeroAttr(eltTy));
        auto reductionAccInit =
            builder.create<tensor::SplatOp>(loc, cst0, reductionAccTy);

        // this is the reduction loop
        SmallVector<Value, 1> reductionResult = createNestedAffineForLoops(
            builder, loc, {rDim}, {r}, reductionAccInit->getResults(),
            [&](OpBuilder &builder, Location loc, ValueRange indices,
                ValueRange iterArgs) -> SmallVector<Value> {
              const auto indexInRedDim = indices[0];

              const ArrayRef<int64_t> lhsSizes{p0, r};

              const Type lhsSliceType = RankedTensorType::get({p0, r}, eltTy);

              const Value lhsSlice = builder.create<tensor::ExtractSliceOp>(
                  loc, lhsSliceType, lhs,
                  ValueRange{parIndices[0], indexInRedDim}, ValueRange{},
                  ValueRange{}, noStaticOffsets, lhsSizes, unitStrides);

              const Type rhsSliceType = RankedTensorType::get({r, p1}, eltTy);

              const ArrayRef<int64_t> rhsSizes{r, p1};

              const Value rhsSlice = builder.create<tensor::ExtractSliceOp>(
                  loc, rhsSliceType, rhs,
                  ValueRange{indexInRedDim, parIndices[1]}, ValueRange{},
                  ValueRange{}, noStaticOffsets, rhsSizes, unitStrides);

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
            ValueRange{}, ValueRange{}, resultOffsets, resultSizes,
            unitStrides);
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
