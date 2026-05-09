
#include <cinm-mlir/Utils/CinmUtils.h>
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineExprVisitor.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <optional>

namespace mlir {

SmallString<20> getUniqueFunctionName(ModuleOp &moduleOp, StringRef prefix) {
  // Note: here we don't use SymbolTable as we run into a bug in the LLVM
  // conversion. Old memref.globals are not cleaned up in time, and for a while
  // the memref.global and llvm.mlir.global exist in the module with the same name.
  // Then SymbolTable cannot be created because names are not unique.
  std::set<StringRef> usedNames;
  for (auto &block : moduleOp.getBodyRegion()) {
    for (auto &op : block) {
      if (auto sym = llvm::dyn_cast_or_null<SymbolOpInterface>(op)) {
        usedNames.insert(sym.getNameAttr());
      }
    }
  }

  // Get a unique global name.
  unsigned stringNumber = 0;
  size_t prefixLen = prefix.size();
  assert(20 > 3 + prefixLen); // make sure this is bigger than the prefix
                              // (prefixes are literals)
  SmallString<20> name(prefix);
  do {
    name.truncate(prefixLen);
    name.append(std::to_string(stringNumber++));
  } while (usedNames.contains(name));
  return name;
}

/// Check that the memref is contiguous in the dimensions corresponding to the
/// bufShape, which is a suffix of the shape of the input tensor/memref.
bool scatteredMemrefIsContiguous(TypedValue<ShapedType> value,
                                 llvm::ArrayRef<int64_t> bufShape) {
  if (isa<MemRefType>(value.getType())) {
    auto type = cast<MemRefType>(value.getType());
    if (!type.hasStaticShape())
      return false;

    SmallVector<int64_t> strides;
    int64_t offset; // offset may be dynamic, we don't
    if (failed(type.getStridesAndOffset(strides, offset)))
      return false;

    // MemRef is contiguous if outer dimensions are size-1 and inner
    // dimensions have unit strides.
    int64_t runningStride = 1;
    int64_t curDim = strides.size() - 1;
    int64_t lastDimToCheck = strides.size() - bufShape.size();
    // Finds all inner dimensions with unit strides.
    while (curDim >= lastDimToCheck && strides[curDim] == runningStride) {
      runningStride *= type.getDimSize(curDim);
      --curDim;
    }

    // Check if other dimensions are size-1.
    while (curDim >= lastDimToCheck && type.getDimSize(curDim) == 1) {
      --curDim;
    }

    // All dims are unit-strided or size-1.
    return curDim < lastDimToCheck;
  }
  return true;
}

/// Simplify the affine expression by flattening it and reconstructing it.
static AffineExpr simplifyAffineExprWithBounds(
    AffineExpr expr, unsigned numDims, unsigned numSymbols,
    llvm::ArrayRef<std::optional<int64_t>> dimLowerBounds,
    llvm::ArrayRef<std::optional<int64_t>> dimUpperBounds) {
  auto kind = expr.getKind();
  switch (kind) {
  case AffineExprKind::Constant:
  case AffineExprKind::SymbolId:
    return expr;
  case AffineExprKind::DimId: {
    auto dimExpr = cast<AffineDimExpr>(expr);
    auto ub = dimUpperBounds[dimExpr.getPosition()];
    auto lb = dimLowerBounds[dimExpr.getPosition()];
    if (ub == lb && ub.has_value() && lb.has_value())
      return getAffineConstantExpr(lb.value(), expr.getContext());
    return dimExpr;
  }
  case AffineExprKind::Add:
  case AffineExprKind::Mul: {
    AffineBinaryOpExpr binaryExpr = cast<AffineBinaryOpExpr>(expr);
    return getAffineBinaryOpExpr(
        expr.getKind(),
        simplifyAffineExprWithBounds(binaryExpr.getLHS(), numDims, numSymbols,
                                     dimLowerBounds, dimUpperBounds),
        simplifyAffineExprWithBounds(binaryExpr.getRHS(), numDims, numSymbols,
                                     dimLowerBounds, dimUpperBounds));
  }
  case AffineExprKind::FloorDiv:
  case AffineExprKind::CeilDiv:
  case AffineExprKind::Mod: {
    AffineBinaryOpExpr binaryExpr = cast<AffineBinaryOpExpr>(expr);
    AffineExpr sLHS =
        simplifyAffineExprWithBounds(binaryExpr.getLHS(), numDims, numSymbols,
                                     dimLowerBounds, dimUpperBounds);
    AffineExpr sRHS =
        simplifyAffineExprWithBounds(binaryExpr.getRHS(), numDims, numSymbols,
                                     dimLowerBounds, dimUpperBounds);

    // We care about the patterns where
    // - we divide by a number which is larger than the upper bound (-> 0)
    // - we do modulo with a number that is greater than the bound of the
    // scrutinee

    auto lhsUB = getBoundForAffineExpr(sLHS, numDims, numSymbols,
                                       dimLowerBounds, dimUpperBounds, true);
    if (auto constRhs = llvm::dyn_cast_or_null<AffineConstantExpr>(sRHS)) {
      auto rhs = constRhs.getValue();
      if (lhsUB) {
        if (kind == AffineExprKind::Mod && *lhsUB < rhs) {
          return sLHS;
        } else if (kind == AffineExprKind::FloorDiv && *lhsUB < rhs) {
          return getAffineConstantExpr(0, expr.getContext());
        }
      }
    }
    return getAffineBinaryOpExpr(kind, sLHS, sRHS);
  }
  }
  llvm_unreachable("Unknown AffineExpr");
}

AffineMap simplifyAffineMapWithBounds(AffineMap map,
                                      llvm::ArrayRef<int64_t> dimSizes) {
  llvm::SmallVector<std::optional<int64_t>> upperBounds;
  for (auto dim : dimSizes) {
    if (dim == ShapedType::kDynamic)
      upperBounds.push_back(std::nullopt);
    else
      upperBounds.push_back(std::make_optional(dim - 1));
  }

  llvm::SmallVector<std::optional<int64_t>> lowerBounds;
  for (auto dim : dimSizes) {
    (void)dim;
    lowerBounds.push_back(std::make_optional(0));
  }

  SmallVector<AffineExpr, 8> exprs;
  for (auto e : map.getResults()) {
    e = simplifyAffineExpr(e, map.getNumDims(), map.getNumSymbols());
    e = simplifyAffineExprWithBounds(e, map.getNumDims(), map.getNumSymbols(),
                                     lowerBounds, upperBounds);
    exprs.push_back(
        simplifyAffineExpr(e, map.getNumDims(), map.getNumSymbols()));
  }
  return AffineMap::get(map.getNumDims(), map.getNumSymbols(), exprs,
                        map.getContext());
}

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
TypedValue<ShapedType> reshapeStatic(OpBuilder &b, Location loc,
                                     TypedValue<ShapedType> value,
                                     llvm::ArrayRef<int64_t> newShape) {
  return reshapeStatic(b, loc, value, value.getType(), newShape);
}

TypedValue<ShapedType> reshapeStatic(OpBuilder &builder, Location loc,
                                     Value value, ShapedType type,
                                     llvm::ArrayRef<int64_t> newShape) {
  auto newTy = type.cloneWith(newShape, type.getElementType());
  auto reifiedShape = builder.create<arith::ConstantOp>(
      loc, RankedTensorType::get({newTy.getRank()}, builder.getI64Type()),
      builder.getI64TensorAttr(newShape));

  if (isa<RankedTensorType>(newTy)) {
    return dyn_cast<TypedValue<ShapedType>>(
        builder.create<tensor::ReshapeOp>(loc, newTy, value, reifiedShape)
            .getResult());
  } else if (isa<MemRefType>(newTy)) {
    return dyn_cast<TypedValue<ShapedType>>(
        builder.create<memref::ReshapeOp>(loc, newTy, value, reifiedShape)
            .getResult());
  }
  assert(false && "must be memref or tensor");
}

} // namespace mlir