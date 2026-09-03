

#include "cinm-mlir/Dialect/Cinm/IR/CinmUtils.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"

#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/CastInterfaces.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Interfaces/ViewLikeInterface.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/RegionUtils.h>

namespace mlir::cinm {

/// Depth budget for the operand recursion below. A padded weight is three
/// steps deep; a chain long enough to exhaust this is answered
/// conservatively (not static) rather than walked, which costs an
/// amortisation opportunity and never correctness.
static constexpr unsigned kStaticValueDepth = 24;

static bool isStaticValueImpl(Value value, unsigned depth) {
  if (depth == 0)
    return false;
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

    // A conclusion someone else already drew. A buffer is filled by an op that
    // writes it rather than produced by one, so nothing reachable from here
    // says where its contents came from -- and the pass that does know may sit
    // in a dialect this library cannot name. It records the answer on the
    // allocation instead, which is the one place both sides can see.
    if (def && def->hasAttr(CinmDialect::STATIC_ATTR_NAME))
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
    if (llvm::isa_and_nonnull<bufferization::ToBufferOp,
                              bufferization::ToTensorOp, CastOpInterface,
                              tensor::ExpandShapeOp, tensor::CollapseShapeOp,
                              tensor::ReshapeOp, memref::CollapseShapeOp,
                              memref::ReshapeOp, memref::ExpandShapeOp>(def)) {
      value = def->getOperand(0);
      continue;
    }

    // A compute block's i-th result is its terminator's i-th yielded value.
    // The dual of the block-argument case above: together they let the walk
    // cross a block in either direction, which is what a value that was
    // computed inside one -- a padded weight, a pre-scaled one -- needs.
    if (auto block = llvm::dyn_cast<ComputeBlockOp>(def)) {
      value = block.getBody().front().getTerminator()->getOperand(
          llvm::cast<OpResult>(value).getResultNumber());
      continue;
    }

    // memref.get_global hands out a *reference*: the op is pure, but the
    // memory behind it holds whatever was last written there -- a repack
    // staging buffer is refilled on every call. So it is static only when
    // the global is a true constant; a mutable global's staticness is the
    // filling pass's conclusion to draw, which it records as the attribute
    // the check above already honours (EnsureScatterGatherContiguous stamps
    // the get_global of a repack whose source is static).
    if (auto getGlobal = llvm::dyn_cast_or_null<memref::GetGlobalOp>(def)) {
      auto global = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
          def, getGlobal.getNameAttr());
      return global && global.getConstant();
    }

    // General rule, and the one the cases above are fast paths for: a pure
    // op applied to static operands yields a static result, since "static"
    // means "does not vary between inferences" and a pure op is a
    // deterministic function of its inputs. This is what sees through the
    // lowering of tensor.pad -- insert_slice(weight, into: constant fill) --
    // which no single-operand walk can follow, and through any other
    // precomputation over weights.
    //
    // Only for value-semantics results: a pure op returning a memref returns
    // a reference, and purity says nothing about the memory behind it (see
    // get_global above). Reference-typed chains are covered by the explicit
    // view/cast cases, the recorded-attribute check, and the block-argument
    // walk -- never by this rule.
    //
    // Regions are part of the input: a body may capture values from above
    // (the fill constant, but equally something per-inference), so those are
    // checked too. An op with neither operands nor captures is static
    // vacuously, which is the right answer for tensor.empty: uninitialised
    // contents do not vary with the inference either.
    if (def && mlir::isMemoryEffectFree(def) &&
        llvm::none_of(def->getResultTypes(),
                      [](Type t) { return llvm::isa<BaseMemRefType>(t); })) {
      llvm::SetVector<Value> captured;
      if (def->getNumRegions() > 0)
        mlir::getUsedValuesDefinedAbove(def->getRegions(), captured);
      auto stillStatic = [&](Value operand) {
        return isStaticValueImpl(operand, depth - 1);
      };
      return llvm::all_of(def->getOperands(), stillStatic) &&
             llvm::all_of(captured, stillStatic);
    }
    return false;
  }
}

bool isStaticValue(Value value) {
  return isStaticValueImpl(value, kStaticValueDepth);
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
