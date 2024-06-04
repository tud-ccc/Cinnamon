#include "cinm-mlir/Dialect/Cinm/Interfaces/TilingInterface.h"

#include <limits>

#include <llvm/ADT/STLExtras.h>

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::cinm;

//===- Generated implementation -------------------------------------------===//

#include "cinm-mlir/Dialect/Cinm/Interfaces/TilingInterface.cpp.inc"

//===----------------------------------------------------------------------===//

namespace mlir::cinm {
Value createVectorReduce2(OpBuilder &builder, Location loc, Value v0, Value v1,
                          Attribute init, ReduceAccumulatorCallback merge2,
                          ReduceAccumulatorCallback reduce,
                          int64_t clusterSize) {
  const RankedTensorType vectorType = v0.getType().cast<RankedTensorType>();
  assert(vectorType.getRank() == 1);
  assert(vectorType.getShape() ==
         v1.getType().cast<RankedTensorType>().getShape());
  const int64_t vectorSize = vectorType.getDimSize(0);
  const Type elementType = vectorType.getElementType();

  const Value initTensor = builder.create<arith::ConstantOp>(
      loc,
      DenseElementsAttr::get(RankedTensorType::get({}, elementType), init));
  Value stage1Output;
  if (clusterSize > 1) {
    const int64_t clusterCount = vectorSize / clusterSize;
    RankedTensorType intermediateResultType =
        RankedTensorType::get(SmallVector<int64_t>{clusterCount}, elementType);
    RankedTensorType reshapedTy = RankedTensorType::get(
        SmallVector<int64_t>{clusterCount, clusterSize}, elementType);

    const Value outTensor = builder.create<arith::ConstantOp>(
        loc, DenseElementsAttr::get(intermediateResultType, init));

    // reified shape clusterCount x clusterSize
    auto shapeTensor = builder.create<arith::ConstantOp>(
        loc, DenseIntElementsAttr::get(
                 RankedTensorType::get({2}, builder.getI64Type()),
                 {clusterCount, clusterSize}));

    auto reshape0 =
        builder.create<tensor::ReshapeOp>(loc, reshapedTy, v0, shapeTensor);
    auto reshape1 =
        builder.create<tensor::ReshapeOp>(loc, reshapedTy, v1, shapeTensor);

    linalg::ReduceOp reduceOp = builder.create<linalg::ReduceOp>(
        loc, ValueRange{reshape0, reshape1}, ValueRange{outTensor},
        ArrayRef<int64_t>{1},
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          const Value tmp = merge2(builder, loc, args[0], args[1]);
          const Value result = reduce(builder, loc, tmp, args[2]);
          builder.create<linalg::YieldOp>(loc, result);
        });
    stage1Output = reduceOp.getResults()[0];
  }

  linalg::ReduceOp sum = builder.create<linalg::ReduceOp>(
      loc, ValueRange{stage1Output}, ValueRange{initTensor},
      SmallVector<int64_t>{0},
      [&](OpBuilder &builder, Location loc, ValueRange args) {
        const Value result = reduce(builder, loc, args[0], args[1]);
        builder.create<linalg::YieldOp>(loc, result);
      });

  return builder.create<tensor::ExtractOp>(loc, sum.getResult(0), ValueRange{});
}

Value createVectorReduce(OpBuilder &builder, Location loc, Value vector,
                         Value init, ReduceAccumulatorCallback callback,
                         int64_t clusterSize) {
  const RankedTensorType vectorType = vector.getType().cast<RankedTensorType>();
  assert(vectorType.getRank() == 1);
  const int64_t vectorSize = vectorType.getDimSize(0);
  const Type elementType = vectorType.getElementType();

  const Value initTensor = builder.create<tensor::FromElementsOp>(
      loc, RankedTensorType::get({}, elementType), ValueRange{init});
  const Value clusterSizeConst =
      builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(clusterSize));

  if (clusterSize > 1) {
    const int64_t clusterCount = vectorSize / clusterSize;
    RankedTensorType intermediateResultType =
        RankedTensorType::get(SmallVector<int64_t>{clusterCount}, elementType);
    vector = builder.create<tensor::GenerateOp>(
        loc, intermediateResultType, ValueRange{},
        [&](OpBuilder &builder, Location loc, ValueRange indices) {
          const SmallVector<int64_t> offsets{ShapedType::kDynamic};
          const SmallVector<int64_t> sizes{clusterSize};
          const SmallVector<int64_t> strides{1};

          const Value dynOffset =
              builder.create<arith::MulIOp>(loc, indices[0], clusterSizeConst);

          const Type sliceType = tensor::ExtractSliceOp::inferResultType(
              vectorType, offsets, sizes, strides);
          const Value slice = builder.create<tensor::ExtractSliceOp>(
              loc, sliceType, vector, ValueRange{dynOffset}, ValueRange{},
              ValueRange{}, offsets, sizes, strides);

          linalg::ReduceOp sum = builder.create<linalg::ReduceOp>(
              loc, ValueRange{slice}, ValueRange{initTensor},
              SmallVector<int64_t>{0},
              [&](OpBuilder &builder, Location loc, ValueRange args) {
                const Value result = callback(builder, loc, args[0], args[1]);
                builder.create<linalg::YieldOp>(loc, result);
              });

          builder.create<tensor::YieldOp>(
              loc, builder.create<tensor::ExtractOp>(loc, sum.getResult(0),
                                                     ValueRange{}));
        });
  }

  linalg::ReduceOp sum = builder.create<linalg::ReduceOp>(
      loc, ValueRange{vector}, ValueRange{initTensor}, SmallVector<int64_t>{0},
      [&](OpBuilder &builder, Location loc, ValueRange args) {
        const Value result = callback(builder, loc, args[0], args[1]);
        builder.create<linalg::YieldOp>(loc, result);
      });

  return builder.create<tensor::ExtractOp>(loc, sum.getResult(0), ValueRange{});
}

Value createVectorReduceAdd(OpBuilder &builder, Location loc, Value vector,
                            int64_t clusterSize) {
  const Type elementType =
      vector.getType().cast<RankedTensorType>().getElementType();
  const Value init = builder.create<arith::ConstantOp>(
      loc, builder.getIntegerAttr(elementType, 0));
  return createVectorReduce<arith::AddIOp>(builder, loc, vector, init,
                                           clusterSize);
}

Value createVectorReduceMul(OpBuilder &builder, Location loc, Value vector,
                            int64_t clusterSize) {
  const Type elementType =
      vector.getType().cast<RankedTensorType>().getElementType();
  const Value init = builder.create<arith::ConstantOp>(
      loc, builder.getIntegerAttr(elementType, 1));
  return createVectorReduce<arith::MulIOp>(builder, loc, vector, init,
                                           clusterSize);
}

Value createVectorReduceMin(OpBuilder &builder, Location loc, Value vector,
                            int64_t clusterSize) {
  const Type elementType =
      vector.getType().cast<RankedTensorType>().getElementType();
  const Value init = builder.create<arith::ConstantOp>(
      loc, builder.getIntegerAttr(elementType,
                                  std::numeric_limits<uint64_t>::max()));
  return createVectorReduce<arith::MinUIOp>(builder, loc, vector, init,
                                            clusterSize);
}

Value createVectorReduceMax(OpBuilder &builder, Location loc, Value vector,
                            int64_t clusterSize) {
  const Type elementType =
      vector.getType().cast<RankedTensorType>().getElementType();
  const Value init = builder.create<arith::ConstantOp>(
      loc, builder.getIntegerAttr(elementType,
                                  std::numeric_limits<uint64_t>::min()));
  return createVectorReduce<arith::MaxUIOp>(builder, loc, vector, init,
                                            clusterSize);
}

Value createMatmul(OpBuilder builder, Location loc, Value lhs, Value rhs,
                   int64_t reduceClusterSize) {
  const RankedTensorType lhsType = lhs.getType().cast<RankedTensorType>();
  const RankedTensorType rhsType = rhs.getType().cast<RankedTensorType>();

  assert(lhsType.getElementType() == rhsType.getElementType());
  const Type elementType = lhsType.getElementType();

  RankedTensorType resultType;
  if (lhsType.getRank() == 2 && rhsType.getRank() == 2) {
    assert(lhsType.getDimSize(1) == rhsType.getDimSize(0));
    resultType = RankedTensorType::get(
        SmallVector<int64_t, 2>{lhsType.getDimSize(0), rhsType.getDimSize(1)},
        elementType);
  } else if (lhsType.getRank() == 1 && rhsType.getRank() == 2) {
    assert(lhsType.getDimSize(0) == rhsType.getDimSize(0));
    resultType = RankedTensorType::get(
        SmallVector<int64_t, 1>{rhsType.getDimSize(1)}, elementType);
  } else if (lhsType.getRank() == 2 && rhsType.getRank() == 1) {
    assert(lhsType.getDimSize(1) == rhsType.getDimSize(0));
    resultType = RankedTensorType::get(
        SmallVector<int64_t, 1>{lhsType.getDimSize(0)}, elementType);
  } else {
    assert(false);
  }

  return builder.create<tensor::GenerateOp>(
      loc, resultType, ValueRange{},
      [&](OpBuilder &builder, Location loc, ValueRange indices) {
        Value lhsSlice = lhs, rhsSlice = rhs;
        const SmallVector<int64_t> strides{1, 1};

        if (lhsType.getRank() == 2) {
          const SmallVector<int64_t> lhsOffsets{ShapedType::kDynamic, 0};
          const SmallVector<int64_t> lhsSizes{1, lhsType.getDimSize(1)};
          const SmallVector<Value> lhsDynamicOffsets{indices.front()};
          const RankedTensorType lhsSliceType = RankedTensorType::get(
              {lhsType.getDimSize(1)}, lhsType.getElementType());
          lhsSlice = builder.create<tensor::ExtractSliceOp>(
              loc, lhsSliceType, lhs, lhsDynamicOffsets, ValueRange{},
              ValueRange{}, lhsOffsets, lhsSizes, strides);
        }

        if (rhsType.getRank() == 2) {
          const SmallVector<int64_t> rhsOffsets{0, ShapedType::kDynamic};
          const SmallVector<int64_t> rhsSizes{rhsType.getDimSize(0), 1};
          const SmallVector<Value> rhsDynamicOffsets{indices.back()};
          const RankedTensorType rhsSliceType = RankedTensorType::get(
              {lhsType.getDimSize(1)}, lhsType.getElementType());
          rhsSlice = builder.create<tensor::ExtractSliceOp>(
              loc, rhsSliceType, rhs, rhsDynamicOffsets, ValueRange{},
              ValueRange{}, rhsOffsets, rhsSizes, strides);
        }

        const Type elementType = lhsType.getElementType();
        const Attribute init = builder.getIntegerAttr(elementType, 0);

        Value result;
        if (elementType.isa<IntegerType>()) {
          result = createVectorReduce2<arith::MulIOp, arith::AddIOp>(
              builder, loc, lhsSlice, rhsSlice, init, reduceClusterSize);
        } else {
          result = createVectorReduce2<arith::MulFOp, arith::AddFOp>(
              builder, loc, lhsSlice, rhsSlice, init, reduceClusterSize);
        }

        builder.create<tensor::YieldOp>(loc, result);
      });
}

SmallVector<int64_t, 2> getTileSizes(ArrayRef<int64_t> tileCounts,
                                     ArrayRef<int64_t> tensorShape) {
  SmallVector<int64_t, 2> tileSizes{tensorShape};
  for (uint64_t i = 0; i < tileSizes.size(); i++) {
    assert(tileSizes[i] % tileCounts[i] == 0);
    tileSizes[i] /= tileCounts[i];
  }
  return tileSizes;
}

} // namespace mlir::cinm
