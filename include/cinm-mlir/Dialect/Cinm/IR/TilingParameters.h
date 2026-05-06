#pragma once

#include "mlir/IR/Builders.h"

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

} // namespace mlir::cinm
