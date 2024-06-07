#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <llvm/ADT/ArrayRef.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <numeric>


namespace mlir::cinm {
struct ComputeOp;
struct TilingParameters {
  /// Maximum size of all buffers inside a compute kernel.
  const int64_t maxBufferSizeInBytes;
  /// Workgroup shape being considered for tiling
  const llvm::ArrayRef<int64_t> workgroupShape;

  TilingParameters(int64_t maxBufferSizeInBytes,
                   llvm::ArrayRef<int64_t> workgroupShape)
      : maxBufferSizeInBytes(maxBufferSizeInBytes),
        workgroupShape(workgroupShape) {}

  /// Return the size of tiles on a reduce dimension.
  /// Computes this by assuming the reduction operation needs (maybe several)
  /// buffers of the same size, same element type. The returned tile size
  /// divides the number of reduced elements.
  int64_t reduceClusterSize(int64_t numBuffers, int64_t reducedElements,
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

  int64_t parallelClusterSize(int64_t parallelElements, int64_t reduceClusterSize) {
    /// need to find a number that divides parallelElements and the working group size
    return std::gcd(parallelElements, workingGroupSize());
  }


  /// Number of parallel elements in the working group.
  int64_t workingGroupSize() {
    return std::reduce(workgroupShape.begin(), workgroupShape.end(), 1,
                       std::multiplies<>());
  }

  int64_t maxNumElementsOfType(Type ty) {
    int64_t bw = std::max(8, static_cast<int>(ty.getIntOrFloatBitWidth()));
    return maxBufferSizeInBytes / (bw / 8);
  }

  static TilingParameters fromComputeBlock(cinm::ComputeOp&);
};

void markOpAsNoTile(Operation*);

using ReduceAccumulatorCallback =
    function_ref<Value(OpBuilder &, Location, Value, Value)>;

template <typename ReductionOp>
Value createVectorReduce(OpBuilder &builder, Location loc, Value vector,
                         Value init, int64_t clusterSize) {
  return createVectorReduce(
      builder, loc, vector, init,
      [](OpBuilder &builder, Location loc, Value lhs, Value rhs) {
        return builder.create<ReductionOp>(loc, lhs, rhs);
      },
      clusterSize);
}

template <typename MergeOp, typename ReductionOp>
Value createVectorReduce2(OpBuilder &builder, Location loc, Value vector,
                          Value vector2, Attribute init, int64_t clusterSize) {
  return createVectorReduce2(
      builder, loc, vector, vector2, init,
      [](OpBuilder &builder, Location loc, Value lhs, Value rhs) {
        return builder.create<MergeOp>(loc, lhs, rhs);
      },
      [](OpBuilder &builder, Location loc, Value lhs, Value rhs) {
        return builder.create<ReductionOp>(loc, lhs, rhs);
      },
      clusterSize);
}

Value createVectorReduce(OpBuilder &builder, Location loc, Value vector,
                         Value init, ReduceAccumulatorCallback callback,
                         int64_t clusterSize = 1);

Value createVectorReduce2(OpBuilder &builder, Location loc, Value vector,
                          Value vector2, Attribute init,
                          ReduceAccumulatorCallback merge2,
                          ReduceAccumulatorCallback reduce,
                          int64_t clusterSize = 1);

Value createVectorReduceAdd(OpBuilder &builder, Location loc, Value vector,
                            int64_t clusterSize = 1);

Value createVectorReduceMul(OpBuilder &builder, Location loc, Value vector,
                            int64_t clusterSize = 1);

Value createVectorReduceMin(OpBuilder &builder, Location loc, Value vector,
                            int64_t clusterSize = 1);

Value createVectorReduceMax(OpBuilder &builder, Location loc, Value vector,
                            int64_t clusterSize = 1);

Value createMatmul(OpBuilder builder, Location loc, Value lhs, Value rhs,
                   int64_t reduceClusterSize = 1);

SmallVector<int64_t, 2> getTileSizes(ArrayRef<int64_t> tileCounts,
                                     ArrayRef<int64_t> tensorShape);

} // namespace mlir::cinm

//===- Generated includes -------------------------------------------------===//

#include "cinm-mlir/Dialect/Cinm/Interfaces/TilingInterface.h.inc"

//===----------------------------------------------------------------------===//