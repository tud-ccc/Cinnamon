//===- CnmScatterMap.cpp - Interpreting cnm.scatter/gather maps ----------===//

#include "cinm-mlir/Dialect/Cnm/IR/CnmScatterMap.h"
#include "cinm-mlir/Utils/CinmUtils.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/AffineExpr.h>

#include <algorithm>
#include <limits>

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

AffineMap mlir::cnm::deflateScatterMap(AffineMap map, BufferType buffer,
                                       ShapedType hostTy) {
  ArrayRef<int64_t> bufShape = buffer.getShape();
  ArrayRef<int64_t> hostShape = hostTy.getShape();
  int64_t wgRank = buffer.getWorkgroupShape().size();
  int64_t numDims = map.getNumDims();
  // The printer calls this, so it has to cope with IR that does not verify.
  if (numDims < wgRank ||
      numDims - wgRank > static_cast<int64_t>(bufShape.size()))
    return map;
  int64_t retained = numDims - wgRank;
  if (static_cast<int64_t>(map.getNumResults()) +
          static_cast<int64_t>(bufShape.size()) - retained !=
      static_cast<int64_t>(hostShape.size()))
    return map;

  // A memref may store a sub-array with gaps in it, and a block that spans a
  // gap is not one run; a tensor has no such thing.
  int64_t contiguous = std::numeric_limits<int64_t>::max();
  if (auto memrefTy = dyn_cast<MemRefType>(hostTy)) {
    contiguous = getContiguousSuffixSize(memrefTy);
    if (contiguous < 0)
      return map;
  }
  int64_t blockElements = computeProduct(getScatterBlockShape(map, buffer));

  SmallVector<AffineExpr> results(map.getResults());
  while (retained > 0 && !results.empty()) {
    int64_t extent = bufShape[retained - 1];
    if (extent != hostShape[results.size() - 1])
      break;
    if (blockElements * extent > contiguous)
      break;
    unsigned dim = wgRank + retained - 1;
    if (results.back() != getAffineDimExpr(dim, map.getContext()))
      break;
    // Dropping the dimension from the domain is only sound if nothing else
    // names it.
    if (llvm::any_of(ArrayRef(results).drop_back(), [&](AffineExpr expr) {
          return expr.isFunctionOfDim(dim);
        }))
      break;
    results.pop_back();
    blockElements *= extent;
    --retained;
  }
  return AffineMap::get(wgRank + retained, map.getNumSymbols(), results,
                        map.getContext());
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
