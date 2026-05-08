#include "cinm-mlir/Dialect/Cinm/IR/TilingInterface.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <numeric>

#include <llvm/ADT/STLExtras.h>

#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LogicalResult.h>
#include <optional>
#include <utility>

using namespace mlir;
using namespace mlir::cinm;

//===- Generated implementation -------------------------------------------===//

#include "cinm-mlir/Dialect/Cinm/IR/TilingInterface.cpp.inc"

//===----------------------------------------------------------------------===//

namespace mlir::cinm {


TypedValue<ShapedType> reshapeStatic(OpBuilder &b, Location loc,
                                     TypedValue<ShapedType> value,
                                     llvm::ArrayRef<int64_t> newShape) {
  return reshapeStatic(b, loc, value, value.getType(), newShape);
}

TypedValue<ShapedType> reshapeStatic(OpBuilder &builder, Location loc,
                                     Value value, ShapedType type,
                                     llvm::ArrayRef<int64_t> newShape) {
  auto newTy = type.cloneWith(newShape, type.getElementType());
  auto reifiedShape = builder.create<arith::ConstantOp>(
      loc, RankedTensorType::get({newTy.getRank()}, builder.getI64Type()),
      builder.getI64TensorAttr(newShape));

  if (isa<RankedTensorType>(newTy)) {
    return dyn_cast<TypedValue<ShapedType>>(
        builder.create<tensor::ReshapeOp>(loc, newTy, value, reifiedShape)
            .getResult());
  } else if (isa<MemRefType>(newTy)) {
    return dyn_cast<TypedValue<ShapedType>>(
        builder.create<memref::ReshapeOp>(loc, newTy, value, reifiedShape)
            .getResult());
  }
  assert(false && "must be memref or tensor");
}

linalg::ReduceOp makeReduceOp(OpBuilder &builder, Location loc, Value input,
                              Value init, int64_t dim,
                              ReduceAccumulatorCallback callback) {
  return builder.create<linalg::ReduceOp>(
      loc, ValueRange{input}, ValueRange{init}, SmallVector<int64_t>{dim},
      [&](OpBuilder &builder, Location loc, ValueRange args) {
        const Value result = callback(builder, loc, args[0], args[1]);
        builder.create<linalg::YieldOp>(loc, result);
      });
}

/// This should probably some util function somewhere.
Value createMLIRValueFromArrayRef(OpBuilder &builder, Location loc,
                                  ArrayRef<int64_t> array) {
  // Define the element type and shape
  auto elementType = builder.getIntegerType(64);
  auto tensorType =
      RankedTensorType::get({static_cast<int64_t>(array.size())}, elementType);

  // Create a DenseElementsAttr with the ArrayRef data
  auto denseAttr = DenseElementsAttr::get(tensorType, array);

  // Create a constant operation to hold the value
  return builder.create<arith::ConstantOp>(loc, tensorType, denseAttr);
}

Value createVectorReduce(OpBuilder &builder, Location loc, Value inputTensor,
                         Value init, ReduceAccumulatorCallback callback,
                         DenseI64ArrayAttr dims, int64_t clusterSize) {
  const auto vectorType = cast<RankedTensorType>(inputTensor.getType());
  assert(dims.size() == 1 &&
         "more than 1 reduction dim is not (yet) supported");
  if (dims.size() != 1) {
    emitError(loc, "More than 1 reduction dimension is not (yet) supported");
  }
  int64_t maybe_neg_dim = dims[0];
  uint64_t dim;
  if (maybe_neg_dim < 0) {
    if (maybe_neg_dim < -vectorType.getRank()) {
      emitError(loc, "Negative reduction dim (" +
                         std::to_string(maybe_neg_dim) +
                         ") must be in [-input rank, 0)");
      assert(false && "negative dim must be in [-input rank, 0)");
    }
    // Indexing from back, e.g. dim=-1 -> rank(2) + dim(-1) = new dim(1)
    dim = vectorType.getRank() + maybe_neg_dim;
  } else {
    dim = maybe_neg_dim;
  }
  if (dim > static_cast<uint64_t>(vectorType.getRank())) {
    emitError(loc, "Reduction dim (" + std::to_string(dim) +
                       ") must be in [0, input rank)");
    assert(false && "dim must be in [0, input rank)");
  }

  const int64_t vectorSize = vectorType.getDimSize(dim);
  const Type elementType = vectorType.getElementType();

  if (clusterSize % vectorType.getShape()[dim] == 0) {
    // TODO: this can be optimized, now we're simply using shape[dim] number of
    // clusters
    // TODO: but we should be using all available clusters for all other dims.
    clusterSize = vectorType.getShape()[dim];
  }

  // Create the init tensor. Shape is all ones with 1 size smaller than input
  // rank.
  const SmallVector<int64_t> initSizes(vectorType.getRank() - 1, 1);
  const Value initTensor = builder.create<tensor::SplatOp>(
      loc, RankedTensorType::get(initSizes, elementType), ValueRange{init});
  const Value clusterSizeConst =
      builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(clusterSize));

  if (clusterSize > 1) {
    const int64_t clusterCount = vectorSize / clusterSize;
    SmallVector<int64_t> tmp(vectorType.getShape());
    tmp[dim] = clusterCount;
    RankedTensorType intermediateResultType =
        RankedTensorType::get(tmp, elementType);

    // Aggregate results of partial reductions using tensor.generate and rewrite
    // to inputTensor. If intermediateResultType[dim] is not 1, we do a second
    // reduction later on. tensor.generate generates 1 element at a time; should
    // yield a scalar.
    inputTensor = builder.create<tensor::GenerateOp>(
        loc, intermediateResultType, ValueRange{},
        [&](OpBuilder &builder, Location loc, const ValueRange indices) {
          const SmallVector<int64_t> offsets(vectorType.getRank(),
                                             ShapedType::kDynamic);
          SmallVector<int64_t> sizes(vectorType.getRank(), 1);
          sizes[dim] = clusterSize;
          const SmallVector<int64_t> strides(vectorType.getRank(), 1);

          // Offsets == indices except we need to multiply the reduction dim
          // index with the size we're reducing
          const Value dynOffset = builder.create<arith::MulIOp>(
              loc, indices[dim], clusterSizeConst);
          ValueRange dynOffsets = indices;
          dynOffsets[dim] = dynOffset;

          const RankedTensorType sliceType =
              tensor::ExtractSliceOp::inferResultType(vectorType, offsets,
                                                      sizes, strides);
          const Value slice = builder.create<tensor::ExtractSliceOp>(
              loc, sliceType, inputTensor, dynOffsets, ValueRange{},
              ValueRange{}, offsets, sizes, strides);

          // FIXME: for reduce with e.g. add, initTensor is added for each
          // cluster
          linalg::ReduceOp intermediate =
              makeReduceOp(builder, loc, slice, initTensor, dim, callback);

          auto constantZero =
              builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(0));
          auto extractIndices = SmallVector<Value>(
              cast<RankedTensorType>(intermediate.getResult(0).getType())
                  .getRank(),
              constantZero);

          // Extract scalar result reducing tensor to scalar.
          builder.create<tensor::YieldOp>(
              loc,
              builder.create<tensor::ExtractOp>(loc, intermediate.getResult(0),
                                                ValueRange{extractIndices}));
        });
  }

  // If ready to return, process tensor shape to correct shape.
  if (cast<RankedTensorType>(inputTensor.getType()).getShape()[dim] == 1) {
    // Reshape (collapse reduction dim) and return.
    // TODO: check simpler implementation that just drops shape[dim] since it's
    // 1 anyway.

    SmallVector<int64_t> outShape(vectorType.getShape().size() - 1, 0);
    for (size_t i = 0; i < vectorType.getShape().size(); i++) {
      if (i < dim) {
        outShape[i] = vectorType.getShape()[i];
      } else if (i > dim) {
        outShape[i - 1] = vectorType.getShape()[i];
      }
    }
    auto shape = createMLIRValueFromArrayRef(builder, loc, outShape);
    auto result = builder.create<tensor::ReshapeOp>(
        loc, RankedTensorType::get(outShape, vectorType.getElementType()),
        inputTensor, shape);

    if (cast<RankedTensorType>(result.getResult().getType()).getRank() == 0) {
      // Extract scalar result of reducing vector to scalar.
      return builder.create<tensor::ExtractOp>(loc, result.getResult(),
                                               ValueRange{});
    } else {
      // Return tensor result as-is
      return result.getResult();
    }
  }

  // Otherwise make a second reduction op (non-tiled)
  // TODO: this step and the previous `if (shape[dim] == 1)` can/should probably
  // be combined. Reduce using linalg builtin reduce op. Also collapses reduced
  // dim for us, e.g. tensor<10x5xf32> -> tensor<10xf32> when reducing on dim=1
  linalg::ReduceOp reduceResult =
      makeReduceOp(builder, loc, inputTensor, initTensor, dim, callback);

  if (cast<RankedTensorType>(reduceResult.getResult(0).getType()).getRank() ==
      0) {
    // Extract scalar result of reducing vector to scalar.
    return builder.create<tensor::ExtractOp>(loc, reduceResult.getResult(0),
                                             ValueRange{});
  } else {
    // Return tensor result as-is
    return reduceResult.getResult(0);
  }
}

Value createVectorReduceAdd(OpBuilder &builder, Location loc, Value vector,
                            const DenseI64ArrayAttr dims, int64_t clusterSize) {
  const Type elementType =
      cast<RankedTensorType>(vector.getType()).getElementType();
  if (FloatType floatType = dyn_cast<FloatType>(elementType)) {
    const TypedAttr zeroAttr = FloatAttr::get(
        elementType, APFloat::getZero(floatType.getFloatSemantics()));
    const Value init = builder.create<arith::ConstantOp>(loc, zeroAttr);
    return createVectorReduce<arith::AddFOp>(builder, loc, vector, init, dims,
                                             clusterSize);
  } else {
    const TypedAttr zeroAttr = IntegerAttr::get(elementType, 0);
    const Value init = builder.create<arith::ConstantOp>(loc, zeroAttr);
    return createVectorReduce<arith::AddIOp>(builder, loc, vector, init, dims,
                                             clusterSize);
  }
}

Value createVectorReduceMul(OpBuilder &builder, Location loc, Value vector,
                            const DenseI64ArrayAttr dims, int64_t clusterSize) {
  const Type elementType =
      cast<RankedTensorType>(vector.getType()).getElementType();
  if (FloatType floatType = dyn_cast<FloatType>(elementType)) {
    const TypedAttr oneAttr =
        FloatAttr::get(elementType, APFloat(floatType.getFloatSemantics(), 1));
    const Value init = builder.create<arith::ConstantOp>(loc, oneAttr);
    return createVectorReduce<arith::MulFOp>(builder, loc, vector, init, dims,
                                             clusterSize);
  } else {
    const TypedAttr oneAttr = IntegerAttr::get(elementType, 1);
    const Value init = builder.create<arith::ConstantOp>(loc, oneAttr);
    return createVectorReduce<arith::MulIOp>(builder, loc, vector, init, dims,
                                             clusterSize);
  }
}

Value createVectorReduceMin(OpBuilder &builder, Location loc, Value vector,
                            const DenseI64ArrayAttr dims, int64_t clusterSize) {
  const Type elementType =
      cast<RankedTensorType>(vector.getType()).getElementType();
  if (FloatType floatType = dyn_cast<FloatType>(elementType)) {
    const TypedAttr maxValAttr = FloatAttr::get(
        elementType, APFloat::getInf(floatType.getFloatSemantics()));
    const Value init = builder.create<arith::ConstantOp>(loc, maxValAttr);
    return createVectorReduce<arith::MinimumFOp>(builder, loc, vector, init,
                                                 dims, clusterSize);
  } else {
    const TypedAttr maxValAttr = IntegerAttr::get(
        elementType,
        APInt::getSignedMaxValue(elementType.getIntOrFloatBitWidth()));
    const Value init = builder.create<arith::ConstantOp>(loc, maxValAttr);
    return createVectorReduce<arith::MinSIOp>(builder, loc, vector, init, dims,
                                              clusterSize);
  }
}

Value createVectorReduceMax(OpBuilder &builder, Location loc, Value vector,
                            const DenseI64ArrayAttr dims, int64_t clusterSize) {
  const Type elementType =
      cast<RankedTensorType>(vector.getType()).getElementType();
  if (FloatType floatType = dyn_cast<FloatType>(elementType)) {
    const TypedAttr minValAttr = FloatAttr::get(
        elementType, -APFloat::getInf(floatType.getFloatSemantics()));
    const Value init = builder.create<arith::ConstantOp>(loc, minValAttr);
    return createVectorReduce<arith::MaximumFOp>(builder, loc, vector, init,
                                                 dims, clusterSize);
  } else {
    const TypedAttr minValAttr = IntegerAttr::get(
        elementType,
        APInt::getSignedMinValue(elementType.getIntOrFloatBitWidth()));
    const Value init = builder.create<arith::ConstantOp>(loc, minValAttr);
    return createVectorReduce<arith::MaxSIOp>(builder, loc, vector, init, dims,
                                              clusterSize);
  }
}

} // namespace mlir::cinm
