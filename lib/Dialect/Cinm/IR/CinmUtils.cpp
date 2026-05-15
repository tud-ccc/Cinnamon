

#include "cinm-mlir/Dialect/Cinm/IR/CinmUtils.h"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/OpDefinition.h>
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
        affine::AffineForOp::create(builder, loc, 0, size, step, iterArgs);
    if (!loops.empty() && !iterArgs.empty()) {
      affine::AffineYieldOp::create(builder, loc, current.getResults());
    }
    loops.push_back(current);
    indices.push_back(current.getRegion().front().getArguments().front());
    iterArgs = current.getRegion().front().getArguments().drop_front();
    builder.setInsertionPointToStart(&current.getRegion().front());
  }

  SmallVector<Value> result = bodyBuilder(builder, loc, indices, iterArgs);
  if (!iterArgs.empty()) {
    affine::AffineYieldOp::create(builder, loc, result);
  }

  builder.setInsertionPointAfter(loops.front());
  return loops.front().getResults();
}

SmallVector<Value> createNestedAffineForLoops(OpBuilder &builder, Location loc,
                                              ArrayRef<OpFoldResult> loopSizes,
                                              ArrayRef<int64_t> loopSteps,
                                              ValueRange iterArgsInit,
                                              BodyBuilderCallback bodyBuilder) {
  assert(loopSizes.size() == loopSteps.size());

  MLIRContext *ctx = builder.getContext();
  // Lower bound is always 0.
  AffineMap zeroMap = AffineMap::getConstantMap(0, ctx);
  // Dynamic upper bound: identity map on one dim operand.
  AffineMap dynUbMap = AffineMap::get(1, 0, getAffineDimExpr(0, ctx));

  SmallVector<affine::AffineForOp> loops;
  SmallVector<Value> indices;
  ValueRange iterArgs = iterArgsInit;

  for (auto [sizeOfr, step] : llvm::zip(loopSizes, loopSteps)) {
    affine::AffineForOp current;
    if (auto staticSize = mlir::getConstantIntValue(sizeOfr)) {
      current = affine::AffineForOp::create(builder, loc, 0, *staticSize, step,
                                                    iterArgs);
    } else {
      Value dynSize = cast<Value>(sizeOfr);
      current = affine::AffineForOp::create(builder, 
          loc, ValueRange{}, zeroMap, ValueRange{dynSize}, dynUbMap, step,
          iterArgs);
    }
    if (!loops.empty() && !iterArgs.empty()) {
      affine::AffineYieldOp::create(builder, loc, current.getResults());
    }
    loops.push_back(current);
    indices.push_back(current.getRegion().front().getArguments().front());
    iterArgs = current.getRegion().front().getArguments().drop_front();
    builder.setInsertionPointToStart(&current.getRegion().front());
  }

  SmallVector<Value> result = bodyBuilder(builder, loc, indices, iterArgs);
  if (!iterArgs.empty()) {
    affine::AffineYieldOp::create(builder, loc, result);
  }

  builder.setInsertionPointAfter(loops.front());
  return loops.front().getResults();
}

} // namespace mlir::cinm
