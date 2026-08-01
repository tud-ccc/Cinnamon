//===- CnmScatterMap.cpp - Interpreting cnm.scatter/gather maps ----------===//

#include "cinm-mlir/Dialect/Cnm/IR/CnmScatterMap.h"
#include "cinm-mlir/Utils/CinmUtils.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/AffineExpr.h>

#include <algorithm>

using namespace mlir;
using namespace mlir::cnm;

int64_t mlir::cnm::getNumRetainedBufferDims(AffineMap map, BufferType buffer) {
  int64_t wgRank = buffer.getWorkgroupShape().size();
  assert(static_cast<int64_t>(map.getNumDims()) >= wgRank &&
         "map does not cover the workgroup");
  return map.getNumDims() - wgRank;
}

bool mlir::cnm::isPointwiseScatterMap(AffineMap map, BufferType buffer) {
  return getNumImplicitHostDims(map, buffer) == 0;
}

ArrayRef<int64_t> mlir::cnm::getScatterBlockShape(AffineMap map,
                                                  BufferType buffer) {
  return buffer.getShape().drop_front(getNumRetainedBufferDims(map, buffer));
}

int64_t mlir::cnm::getScatterBlocksPerLeaf(AffineMap map, BufferType buffer) {
  return computeProduct(
      buffer.getShape().take_front(getNumRetainedBufferDims(map, buffer)));
}

int64_t mlir::cnm::getNumImplicitHostDims(AffineMap map, BufferType buffer) {
  return static_cast<int64_t>(buffer.getShape().size()) -
         getNumRetainedBufferDims(map, buffer);
}

AffineMap mlir::cnm::inflateScatterMapToPointwise(AffineMap map,
                                                  BufferType buffer) {
  int64_t implicit = getNumImplicitHostDims(map, buffer);
  if (implicit == 0)
    return map;

  // The implicit host dimensions are indexed by the implicit buffer
  // dimensions, one for one, which is what makes them a block.
  MLIRContext *ctx = map.getContext();
  SmallVector<AffineExpr> results(map.getResults());
  for (int64_t i = 0; i < implicit; ++i)
    results.push_back(getAffineDimExpr(map.getNumDims() + i, ctx));
  return AffineMap::get(map.getNumDims() + implicit, map.getNumSymbols(),
                        results, ctx);
}

SmallVector<int64_t> mlir::cnm::getScatterIndexSpace(BufferType buffer) {
  SmallVector<int64_t> extents(buffer.getWorkgroupShape());
  llvm::append_range(extents, buffer.getShape());
  return extents;
}

SmallVector<int64_t> mlir::cnm::getScatterMapDomain(AffineMap map,
                                                    BufferType buffer) {
  SmallVector<int64_t> extents = getScatterIndexSpace(buffer);
  extents.truncate(map.getNumDims());
  return extents;
}

FailureOr<AffineExpr>
mlir::cnm::linearizeScatterMap(AffineMap map, ArrayRef<int64_t> hostShape) {
  if (map.getNumResults() != hostShape.size())
    return failure();
  MLIRContext *ctx = map.getContext();
  AffineMap layout =
      AffineMap::get(hostShape.size(), 0, linearizeIndices(ctx, hostShape), ctx);
  // Deliberately not simplified: simplification rewrites `x mod c` as
  // `x - (x floordiv c) * c`, which is the same value but splits one
  // dimension across two correlated terms, and getAffineUpperBound reasons
  // term by term.
  return layout.compose(map).getResult(0);
}
