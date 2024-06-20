#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"

#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <optional>
#include <utility>

namespace mlir::cinm {
struct ComputeOp;
struct TilingParameters {
  /// Workgroup shape being considered for tiling.
  const llvm::ArrayRef<int64_t> workgroupShape;
  /// Sizes of the different buffer levels.
  /// Buffers at one level are shared with later levels.
  /// For instance for a workgroup with shape {A,B,C}
  /// and buffer sizes {M,N,P}, the space available
  /// for each compute leaf is P + N/C + M/B/C.
  const llvm::ArrayRef<int64_t> bufferSizesInBytes;

  TilingParameters(llvm::ArrayRef<int64_t> bufferSizesInBytes,
                   llvm::ArrayRef<int64_t> workgroupShape)
      : workgroupShape(workgroupShape),
        bufferSizesInBytes(bufferSizesInBytes) {}

  /// Return the size of tiles on a reduce dimension.
  /// Computes this by assuming the reduction operation needs (maybe several)
  /// buffers of the same size, same element type. The returned tile size
  /// divides the number of reduced elements.
  int64_t reduceClusterSize(int64_t numBuffers, int64_t reducedElements,
                            Type elementTy, int64_t extraElements = 0);

  /// Determine tiling factors for dimensions n and m.
  std::optional<std::pair<int64_t, int64_t>> parallelClusterSize(int64_t n, int64_t m);

  /// Number of parallel elements in the working group.
  int64_t workingGroupSize();

  int64_t bufferSizeOfLeaf();

  int64_t maxNumElementsOfType(Type ty);

  static TilingParameters fromComputeBlock(cinm::ComputeOp &);
};

/// Exclude a cinm op from the --cinm-tiling pass
void markOpAsNoTile(Operation *);

/// Create a tensor.reshape for a fully static tensor shape
Value reshapeStatic(OpBuilder &, Location loc, Value value, ShapedType type,
                    llvm::ArrayRef<int64_t> newShape);

/// Create a tensor.reshape for a fully static tensor shape
Value reshapeStatic(OpBuilder &b, Location loc,
                    TypedValue<RankedTensorType> value,
                    llvm::ArrayRef<int64_t> newShape);

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

template <typename IntOp, typename FloatOp>
Value createArithIntOrFloatOp(OpBuilder &builder, Location loc, Value a,
                              Value b) {
  assert(a.getType() == b.getType() && "Mismatched type");
  assert(a.getType().isIntOrIndexOrFloat() && "Expected scalar type");
  if (a.getType().isa<IntegerType>()) {
    return builder.create<IntOp>(loc, a, b);
  } else {
    return builder.create<FloatOp>(loc, a, b);
  }
}

inline Value createArithAdd(OpBuilder &builder, Location loc, Value a,
                            Value b) {
  return createArithIntOrFloatOp<arith::AddIOp, arith::AddFOp>(builder, loc, a,
                                                               b);
}

inline Value createArithMul(OpBuilder &builder, Location loc, Value a,
                            Value b) {
  return createArithIntOrFloatOp<arith::MulIOp, arith::MulFOp>(builder, loc, a,
                                                               b);
}

Value createVectorReduce(OpBuilder &builder, Location loc, Value vector,
                         Value init, ReduceAccumulatorCallback callback,
                         int64_t clusterSize = 1);

Value createVectorReduceAdd(OpBuilder &builder, Location loc, Value vector,
                            int64_t clusterSize = 1);

Value createVectorReduceMul(OpBuilder &builder, Location loc, Value vector,
                            int64_t clusterSize = 1);

Value createVectorReduceMin(OpBuilder &builder, Location loc, Value vector,
                            int64_t clusterSize = 1);

Value createVectorReduceMax(OpBuilder &builder, Location loc, Value vector,
                            int64_t clusterSize = 1);

} // namespace mlir::cinm

//===- Generated includes -------------------------------------------------===//

#include "cinm-mlir/Dialect/Cinm/Interfaces/TilingInterface.h.inc"

//===----------------------------------------------------------------------===//