
#include <cinm-mlir/Utils/CinmUtils.h>
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineExprVisitor.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <optional>

namespace mlir {

SmallString<20> getUniqueFunctionName(SymbolTable &moduleOp, StringRef prefix) {
  // Get a unique global name.
  unsigned stringNumber = 0;
  size_t prefixLen = prefix.size();
  assert(20 > 3 + prefixLen); // make sure this is bigger than the prefix
                              // (prefixes are literals)
  SmallString<20> name(prefix);
  do {
    name.truncate(prefixLen);
    name.append(std::to_string(stringNumber++));
  } while (moduleOp.lookup(name));
  return name;
}

/// Check that the memref is contiguous in the dimensions corresponding to the
/// bufShape, which is a suffix of the shape of the input tensor/memref.
bool scatteredMemrefIsContiguous(TypedValue<ShapedType> value,
                                 llvm::ArrayRef<int64_t> bufShape) {
  if (value.getType().isa<MemRefType>()) {
    auto type = value.getType().cast<MemRefType>();
    if (!type.hasStaticShape())
      return false;

    SmallVector<int64_t> strides;
    int64_t offset; // offset may be dynamic, we don't
    if (failed(getStridesAndOffset(type, strides, offset)))
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
    auto bound = dimUpperBounds[dimExpr.getPosition()];
    if (bound == 1)
      return getAffineConstantExpr(0, expr.getContext());
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
    upperBounds.push_back(std::make_optional(0));
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

} // namespace mlir