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
#include <mlir/Support/LLVM.h>
#include <optional>
#include <utility>

namespace mlir::cinm {
struct ComputeOp;

enum class ReductionDimMode : uint8_t {
  Unspecified = 0,
  Fixed = 1,
  Heuristic = 2
};

struct TilingParameters {
  /// Workgroup shape being considered for tiling.
  const ArrayRef<int64_t> workgroupShape;
  /// Sizes of the different buffer levels.
  const ArrayRef<int64_t> bufferSizesInBytes;

  // Opaque tiling sizes provided by the compute block (operation-agnostic).
  // Ordering/meaning is decided by each consuming op.
  std::optional<SmallVector<int64_t, 8>> tileSizes;

  TilingParameters(ArrayRef<int64_t> bufferSizesInBytes,
                   ArrayRef<int64_t> workgroupShape)
      : workgroupShape(workgroupShape), bufferSizesInBytes(bufferSizesInBytes) {
  }

  int64_t reduceClusterSize(int64_t numBuffers, int64_t reducedElements,
                            Type elementTy, int64_t extraElements = 0) const;
  std::optional<std::pair<int64_t, int64_t>>
  parallelClusterSize(int64_t n, int64_t m) const;
  int64_t workingGroupSize() const;
  int64_t bufferSizeOfLeaf() const;
  int64_t maxNumElementsOfType(Type ty) const;

  static TilingParameters fromComputeBlock(cinm::ComputeOp &);

  // helpers to access the opaque tile sizes.
  std::optional<ArrayRef<int64_t>> getTileSizes() const {
    if (!tileSizes)
      return std::nullopt;
    return ArrayRef<int64_t>(*tileSizes);
  }
  std::optional<int64_t> getTileSize(size_t i) const {
    if (!tileSizes || i >= tileSizes->size())
      return std::nullopt;
    return (*tileSizes)[i];
  }
  // explicit tiles if provided by compute.tileSizes
  // Convention for now: [parallel0, parallel1, reduction] for GEMM.
  std::optional<std::pair<int64_t, int64_t>> getProvidedParallelTiles() const {
    if (!tileSizes || tileSizes->size() < 2)
      return std::nullopt;
    return {{(*tileSizes)[0], (*tileSizes)[1]}};
  }
  std::optional<int64_t> getProvidedReductionTile() const {
    if (!tileSizes || tileSizes->size() < 3)
      return std::nullopt;
    return (*tileSizes)[2];
  }
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
                         Value init, DenseI64ArrayAttr dims,
                         int64_t clusterSize) {
  return createVectorReduce(
      builder, loc, vector, init,
      [](OpBuilder &builder, Location loc, Value lhs, Value rhs) {
        return builder.create<ReductionOp>(loc, lhs, rhs);
      },
      dims, clusterSize);
}

template <typename IntOp, typename FloatOp>
Value createArithIntOrFloatOp(OpBuilder &builder, Location loc, Value a,
                              Value b) {
  assert(a.getType() == b.getType() && "Mismatched type");
  assert(a.getType().isIntOrIndexOrFloat() && "Expected scalar type");
  if (isa<IntegerType>(a.getType())) {
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
                         DenseI64ArrayAttr dims, int64_t clusterSize = 1);

Value createVectorReduceAdd(OpBuilder &builder, Location loc, Value vector,
                            DenseI64ArrayAttr dims, int64_t clusterSize = 1);

Value createVectorReduceMul(OpBuilder &builder, Location loc, Value vector,
                            DenseI64ArrayAttr dims, int64_t clusterSize = 1);

Value createVectorReduceMin(OpBuilder &builder, Location loc, Value vector,
                            DenseI64ArrayAttr dims, int64_t clusterSize = 1);

Value createVectorReduceMax(OpBuilder &builder, Location loc, Value vector,
                            DenseI64ArrayAttr dims, int64_t clusterSize = 1);

} // namespace mlir::cinm

//===- Generated includes -------------------------------------------------===//

#include "cinm-mlir/Dialect/Cinm/IR/TilingInterface.h.inc"

//===----------------------------------------------------------------------===//