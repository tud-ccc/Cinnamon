#include "cinm-mlir/Dialect/Cinm/Interfaces/TilingInterface.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <numeric>

#include <llvm/ADT/STLExtras.h>

#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::cinm;

//===- Generated implementation -------------------------------------------===//

#include "cinm-mlir/Dialect/Cinm/Interfaces/TilingInterface.cpp.inc"

//===----------------------------------------------------------------------===//

namespace mlir::cinm {

void markOpAsNoTile(Operation *op) {
  op->setAttr(CinmDialect::NOTILE_NAME, UnitAttr::get(op->getContext()));
}

TilingParameters TilingParameters::fromComputeBlock(cinm::ComputeOp &op) {
  return TilingParameters(op.getMaxDpuBufferSize() * 4, op.getWorkgroupShape());
}

/// Return the size of tiles on a reduce dimension.
/// Computes this by assuming the reduction operation needs (maybe several)
/// buffers of the same size, same element type. The returned tile size
/// divides the number of reduced elements.
int64_t TilingParameters::reduceClusterSize(int64_t numBuffers,
                                            int64_t reducedElements,
                                            Type elementTy) {
  // in number of elements
  auto maxSizePerBuffer = maxNumElementsOfType(elementTy) / numBuffers;
  // Now we need to find the largest divisor of `reducedElements` that is
  // smaller than maxSizePerBuffer
  for (int i = maxSizePerBuffer; i > 0; i--) {
    if (reducedElements % i == 0)
      return i;
  }
  return 1;
}

/// Determine tiling factors for dimensions n and m.
std::pair<int64_t, int64_t> TilingParameters::parallelClusterSize(int64_t n,
                                                                  int64_t m) {
  /// need to find a number that divides parallelElements and the working
  /// group size
  SmallVector<int64_t, 4> wgShape(workgroupShape);
  auto it = std::remove(wgShape.begin(), wgShape.end(), 1);
  wgShape.erase(it);

  if (wgShape.size() == 2) {
    // try to fit perfectly
    if (n % wgShape[0] == 0 && m % wgShape[1] == 0)
      return {wgShape[0], wgShape[1]};
    else if (n % wgShape[1] == 0 && m % wgShape[0] == 0)
      return {wgShape[1], wgShape[0]};
  }

  auto wg = workingGroupSize();
  auto a = std::gcd(n, wg);
  auto b = std::gcd(m, wg);

  if (a * b == wg)
    return {a, b};
  else if (a > b)
    return {a, wg / a};
  else if (b != 1)
    return {wg / b, b};
  else
    // fallback: don't tile. WG is not divisible by n or m.
    // This means the CINM->CNM conversion will fail.
    return {1, 1};
}

/// Number of parallel elements in the working group.
int64_t TilingParameters::workingGroupSize() {
  return std::reduce(workgroupShape.begin(), workgroupShape.end(), 1,
                     std::multiplies<>());
}

int64_t TilingParameters::maxNumElementsOfType(Type ty) {
  int64_t bw = std::max(8, static_cast<int>(ty.getIntOrFloatBitWidth()));
  return maxBufferSizeInBytes / (bw / 8);
}

Value reshapeStatic(OpBuilder &b, Location loc,
                    TypedValue<RankedTensorType> value,
                    llvm::ArrayRef<int64_t> newShape) {
  return reshapeStatic(b, loc, value, value.getType(), newShape);
}

Value reshapeStatic(OpBuilder &builder, Location loc, Value value,
                    ShapedType type, llvm::ArrayRef<int64_t> newShape) {
  auto newTy = type.cloneWith(newShape, type.getElementType());
  if (newTy.isa<RankedTensorType>()) {
    auto reifiedShape = builder.create<arith::ConstantOp>(
        loc, RankedTensorType::get({newTy.getRank()}, builder.getI64Type()),
        builder.getI64TensorAttr(newShape));
    return builder.create<tensor::ReshapeOp>(loc, newTy, value, reifiedShape);
  }
  // todo memref
  assert(false && "not handled for memrefs for now");
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

// todo these impls should work also for float types

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

} // namespace mlir::cinm
