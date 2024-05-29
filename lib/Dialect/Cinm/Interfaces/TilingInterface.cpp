#include "cinm-mlir/Dialect/Cinm/Interfaces/TilingInterface.h"

#include "mlir/IR/OpImplementation.h"
#include <cstdint>
#include <limits>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::cinm;

//===- Generated implementation -------------------------------------------===//

#include "cinm-mlir/Dialect/Cinm/Interfaces/TilingInterface.cpp.inc"

//===----------------------------------------------------------------------===//

namespace mlir::cinm {

SmallVector<Value> createNestedAffineForLoops(OpBuilder &builder, Location loc,
                                              ArrayRef<int64_t> loopSizes,
                                              ValueRange iterArgsInit,
                                              BodyBuilderCallback bodyBuilder) {
  SmallVector<affine::AffineForOp> loops;

  SmallVector<Value> indices;
  ValueRange iterArgs = iterArgsInit;

  for (int64_t size : loopSizes) {
    affine::AffineForOp current =
        builder.create<affine::AffineForOp>(loc, 0, size, 1, iterArgs);
    if (!loops.empty()) {
      builder.create<affine::AffineYieldOp>(loc, current.getResults());
    }
    loops.push_back(current);
    indices.push_back(current.getRegion().front().getArguments().front());
    iterArgs = current.getRegion().front().getArguments().drop_front();
    builder.setInsertionPointToStart(&current.getRegion().front());
  }

  builder.create<affine::AffineYieldOp>(
      loc, bodyBuilder(builder, loc, indices, iterArgs));

  builder.setInsertionPointAfter(loops.front());
  return loops.front().getResults();
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

  if (clusterSize > 1) {
    const int64_t clusterCount = vectorSize / clusterSize;
    RankedTensorType intermediateResultType =
        RankedTensorType::get(SmallVector<int64_t>{clusterCount}, elementType);
    vector = builder.create<tensor::GenerateOp>(
        loc, intermediateResultType, ValueRange{},
        [&](OpBuilder &builder, Location loc, ValueRange indices) {
          SmallVector<int64_t> offsets{ShapedType::kDynamic};
          SmallVector<int64_t> sizes{clusterSize};
          SmallVector<int64_t> strides{clusterSize};
          Type sliceType = tensor::ExtractSliceOp::inferResultType(
              vectorType, offsets, sizes, strides);
          Value slice = builder.create<tensor::ExtractSliceOp>(
              loc, sliceType, vector, indices, ValueRange{}, ValueRange{},
              offsets, sizes, strides);

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
  assert(lhsType.getRank() == 2 && rhsType.getRank() == 2);
  assert(lhsType.getDimSize(1) == rhsType.getDimSize(0));

  const Type elementType = lhsType.getElementType();
  const RankedTensorType resultType = RankedTensorType::get(
      SmallVector<int64_t, 2>{lhsType.getDimSize(0), rhsType.getDimSize(1)},
      elementType);

  return builder.create<tensor::GenerateOp>(
      loc, resultType, ValueRange{},
      [&](OpBuilder &builder, Location loc, ValueRange indices) {
        const SmallVector<int64_t> lhsOffsets{ShapedType::kDynamic, 0};
        const SmallVector<int64_t> rhsOffsets{0, ShapedType::kDynamic};
        const SmallVector<int64_t> lhsSizes{1, lhsType.getDimSize(1)};
        const SmallVector<int64_t> rhsSizes{rhsType.getDimSize(0), 1};
        const SmallVector<int64_t> strides{1, 1};

        SmallVector<Value> lhsDynamicOffsets{indices[0]};
        RankedTensorType lhsSliceType =
            tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
                1, lhsType, lhsOffsets, lhsSizes, strides);
        Value lhsSlice = builder.create<tensor::ExtractSliceOp>(
            loc, lhsSliceType, lhs, lhsDynamicOffsets, ValueRange{},
            ValueRange{}, lhsOffsets, lhsSizes, strides);

        SmallVector<Value> rhsDynamicOffsets{indices[1]};
        Type rhsSliceType =
            tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
                1, rhsType, rhsOffsets, rhsSizes, strides);
        Value rhsSlice = builder.create<tensor::ExtractSliceOp>(
            loc, rhsSliceType, rhs, rhsDynamicOffsets, ValueRange{},
            ValueRange{}, rhsOffsets, rhsSizes, strides);

        const Value product =
            builder.create<arith::MulIOp>(loc, lhsSlice, rhsSlice);
        const Value sum =
            createVectorReduceAdd(builder, loc, product, reduceClusterSize);
        builder.create<tensor::YieldOp>(loc, sum);
      });
}

} // namespace mlir::cinm
