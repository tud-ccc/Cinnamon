

#include "cinm-mlir/Dialect/Cinm/IR/CinmUtils.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Interfaces/ViewLikeInterface.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::cinm {

bool isStaticValue(Value value) {
  while (true) {
    if (auto arg = llvm::dyn_cast<BlockArgument>(value)) {
      Operation *owner = arg.getOwner()->getParentOp();
      // An isolated compute block's i-th region argument is its i-th operand.
      if (auto block = llvm::dyn_cast<ComputeBlockOp>(owner)) {
        value = block->getOperand(arg.getArgNumber());
        continue;
      }
      auto func = llvm::dyn_cast<FunctionOpInterface>(owner);
      return func &&
             func.getArgAttr(arg.getArgNumber(), CinmDialect::STATIC_ATTR_NAME);
    }

    Operation *def = value.getDefiningOp();
    if (matchPattern(def, m_Constant()))
      return true;

    // A pure slicing op reads one source at offsets/sizes/strides; with all
    // of those static it has exactly one operand, which ODS puts first
    // (`$source` in tensor.extract_slice and memref.subview). The operand
    // count also excludes ops like tensor.insert_slice that share the
    // interface but combine *two* tensors into their result.
    auto view = llvm::dyn_cast_or_null<OffsetSizeAndStrideOpInterface>(def);
    if (view && view->getNumResults() == 1 && view->getNumOperands() == 1 &&
        llvm::none_of(view.getStaticOffsets(), ShapedType::isDynamic) &&
        llvm::none_of(view.getStaticSizes(), ShapedType::isDynamic) &&
        llvm::none_of(view.getStaticStrides(), ShapedType::isDynamic)) {
      value = view->getOperand(0);
      continue;
    }
    return false;
  }
}

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
      current = affine::AffineForOp::create(builder, loc, ValueRange{}, zeroMap,
                                            ValueRange{dynSize}, dynUbMap, step,
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
