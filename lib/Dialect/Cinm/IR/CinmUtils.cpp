

#include "cinm-mlir/Dialect/Cinm/IR/CinmUtils.h"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::cinm {

SmallVector<Value> createNestedAffineForLoops(OpBuilder &builder, Location loc,
                                              ArrayRef<int64_t> loopSizes,
                                              ArrayRef<int64_t> loopSteps,
                                              ValueRange iterArgsInit,
                                              BodyBuilderCallback bodyBuilder) {
  assert(loopSizes.size() == loopSteps.size());

  SmallVector<affine::AffineForOp> loops;
  SmallVector<Value> indices;
  ValueRange iterArgs = iterArgsInit;

  for (auto [size, step] : llvm::zip(loopSizes, loopSteps)) {
    affine::AffineForOp current =
        builder.create<affine::AffineForOp>(loc, 0, size, step, iterArgs);
    if (!loops.empty() && !iterArgs.empty()) {
      builder.create<affine::AffineYieldOp>(loc, current.getResults());
    }
    loops.push_back(current);
    indices.push_back(current.getRegion().front().getArguments().front());
    iterArgs = current.getRegion().front().getArguments().drop_front();
    builder.setInsertionPointToStart(&current.getRegion().front());
  }

  SmallVector<Value> result = bodyBuilder(builder, loc, indices, iterArgs);
  if (!iterArgs.empty()) {
    builder.create<affine::AffineYieldOp>(loc, result);
  }

  builder.setInsertionPointAfter(loops.front());
  return loops.front().getResults();
}

// Turn an index in the index space of the given shape into a linear index.
AffineExpr linearizeIndices(MLIRContext *ctx, ArrayRef<int64_t> shape) {

  AffineExpr index = getAffineConstantExpr(0, ctx);
  int64_t dimIndex = shape.size() - 1;
  int64_t trailing = 1;
  for (auto it = shape.rbegin(); it != shape.rend(); it++) {
    auto dim = *it;
    index = trailing * getAffineDimExpr(dimIndex, ctx) + index;
    trailing *= dim;
    dimIndex--;
  }
  return index;
}

// inflate a linear index into the given shape
void structureIndex(AffineExpr index, ArrayRef<int64_t> shape,
                    SmallVectorImpl<AffineExpr> &map) {

  int64_t sizeOfTrailing = computeProduct(shape) / shape[0];
  map.push_back(index.floorDiv(sizeOfTrailing));

  AffineExpr gatherExpr = index * sizeOfTrailing;
  size_t i = 1;

  for (auto dim : llvm::drop_begin(shape, 1)) {
    index = index % sizeOfTrailing;
    sizeOfTrailing /= dim;
    map.push_back(index.floorDiv(sizeOfTrailing));
    gatherExpr = gatherExpr +
                 mlir::getAffineDimExpr(i, index.getContext()) * sizeOfTrailing;
    i++;
  }
}
} // namespace mlir::cinm
