
#include <cinm-mlir/Utils/CinmUtils.h>
#include <cstdint>
#include <functional>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineExprVisitor.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/Matchers.h>
#include <optional>

namespace mlir {

static bool isZeroAttr(Attribute attr) {
  if (auto intAttr = dyn_cast_or_null<IntegerAttr>(attr))
    return intAttr.getValue().isZero();
  if (auto floatAttr = dyn_cast_or_null<FloatAttr>(attr))
    return floatAttr.getValue().isZero();
  return false;
}

/// The splat element of `attr`, or nullopt if it isn't a splat.
static std::optional<TypedAttr> getSplatElement(DenseElementsAttr attr) {
  if (!attr || !attr.isSplat())
    return std::nullopt;
  if (auto typed = dyn_cast<TypedAttr>(attr.getSplatValue<Attribute>()))
    return typed;
  return std::nullopt;
}

std::optional<TypedAttr> getUniformValue(Value v) {
  if (!isa<ShapedType>(v.getType()))
    return std::nullopt;

  // A splat constant, or anything whose folder produces one.
  DenseElementsAttr dense;
  if (matchPattern(v, m_Constant(&dense)))
    return getSplatElement(dense);

  // `linalg.fill` has no folder producing a constant tensor (upstream
  // deliberately keeps the value lazy and pushes fills through consumers
  // instead), so it has to be matched directly. A fill overwrites every
  // element, so its destination operand is irrelevant here.
  if (auto fill = v.getDefiningOp<linalg::FillOp>()) {
    TypedAttr scalar;
    if (fill.getInputs().size() == 1 &&
        matchPattern(fill.getInputs()[0], m_Constant(&scalar)))
      return scalar;
    return std::nullopt;
  }
  if (auto fill = v.getDefiningOp<linalg::FillOp>()) {
    TypedAttr scalar;
    if (fill.getInputs().size() == 1 &&
        matchPattern(fill.getInputs()[0], m_Constant(&scalar)))
      return scalar;
    return std::nullopt;
  }

  // A constant global with a splat initializer -- what a splat `arith.constant`
  // becomes after bufferization.
  if (auto getGlobal = v.getDefiningOp<memref::GetGlobalOp>()) {
    auto global = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
        getGlobal, getGlobal.getNameAttr());
    if (!global || !global.getConstant())
      return std::nullopt;
    return getSplatElement(
        dyn_cast_or_null<DenseElementsAttr>(global.getInitialValueAttr()));
  }

  return std::nullopt;
}

bool isZeroSplatFoldable(Value v) {
  if (auto uniform = getUniformValue(v))
    return isZeroAttr(*uniform);

  // Slow path: try folding the defining op with whatever constant operands
  // are available (non-constant operands are passed as null Attributes).
  Operation *defOp = v.getDefiningOp();
  if (!defOp || defOp->getNumResults() != 1)
    return false;

  SmallVector<Attribute> foldOperands(defOp->getNumOperands());
  for (auto [i, operand] : llvm::enumerate(defOp->getOperands())) {
    Attribute opAttr;
    if (matchPattern(operand, m_Constant(&opAttr)))
      foldOperands[i] = opAttr;
  }

  SmallVector<OpFoldResult> foldResults;
  if (failed(defOp->fold(foldOperands, foldResults)) || foldResults.size() != 1)
    return false;

  auto splat = getSplatElement(dyn_cast_or_null<DenseElementsAttr>(
      foldResults[0].dyn_cast<Attribute>()));
  return splat && isZeroAttr(*splat);
}

ShapedType asShaped(Type ty) {
  if (auto shaped = llvm::dyn_cast_or_null<ShapedType>(ty))
    return shaped;
  assert(TensorType::isValidElementType(ty) &&
         "expected a shaped or scalar operand type");
  return RankedTensorType::get({}, ty);
}

SmallString<20> getUniqueFunctionName(ModuleOp &moduleOp, StringRef prefix) {
  // Note: here we don't use SymbolTable as we run into a bug in the LLVM
  // conversion. Old memref.globals are not cleaned up in time, and for a while
  // the memref.global and llvm.mlir.global exist in the module with the same
  // name. Then SymbolTable cannot be created because names are not unique.
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

bool memrefIsContiguous(MemRefType ty) {
  SmallVector<int64_t> strides;
  int64_t offset = 0;
  if (failed(ty.getStridesAndOffset(strides, offset)))
    return false;
  int64_t expected = 1;
  for (int i = ty.getRank() - 1; i >= 0; --i) {
    int64_t size = ty.getDimSize(i);
    if (size == 1)
      continue;
    if (ShapedType::isDynamic(size) || ShapedType::isDynamic(strides[i]) ||
        strides[i] != expected)
      return false;
    expected *= size;
  }
  return true;
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

    // MemRef is contiguous if the inner dimensions (corresponding to
    // bufShape) are packed row-major, and the remaining outer dimensions
    // are all size-1 (so they don't introduce any gaps between repeats of
    // the inner block, regardless of their stride).
    int64_t runningStride = 1;
    int64_t curDim = strides.size() - 1;
    int64_t lastDimToCheck = strides.size() - bufShape.size();
    // Finds all inner dimensions with unit strides.
    while (curDim >= lastDimToCheck && strides[curDim] == runningStride) {
      runningStride *= type.getDimSize(curDim);
      --curDim;
    }
    // The inner (bufShape) dimensions must be fully packed: if we stopped
    // before reaching lastDimToCheck, some inner dimension broke contiguity.
    if (curDim >= lastDimToCheck)
      return false;

    // Check that all remaining (outer) dimensions are size-1. Note this
    // must range all the way down to 0, not just down to lastDimToCheck:
    // those outer dims are exactly the ones not covered by bufShape.
    while (curDim >= 0 && type.getDimSize(curDim) == 1) {
      --curDim;
    }

    // All dims are either part of the packed inner block, or size-1.
    return curDim < 0;
  }
  return true;
}

int64_t getContiguousSuffixSize(MemRefType type) {
  if (type.getLayout().isIdentity())
    return type.hasStaticShape() ? type.getNumElements() : -1;

  auto strided = llvm::dyn_cast<StridedLayoutAttr>(type.getLayout());
  if (!strided)
    return -1;

  ArrayRef<int64_t> shape = type.getShape();
  ArrayRef<int64_t> strides = strided.getStrides();
  int64_t expectedStride = 1;
  int64_t count = 1;
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    if (ShapedType::isDynamic(shape[i]) || ShapedType::isDynamic(strides[i]) ||
        strides[i] != expectedStride)
      break;
    count *= shape[i];
    expectedStride *= shape[i];
  }
  return count;
}

int64_t getContiguousSuffixRank(MemRefType type) {
  if (type.getLayout().isIdentity())
    return type.hasStaticShape() ? type.getRank() : -1;

  auto strided = llvm::dyn_cast<StridedLayoutAttr>(type.getLayout());
  if (!strided)
    return -1;

  ArrayRef<int64_t> shape = type.getShape();
  ArrayRef<int64_t> strides = strided.getStrides();
  int64_t expectedStride = 1;
  int64_t rank = 0;
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    if (ShapedType::isDynamic(shape[i]) || ShapedType::isDynamic(strides[i]) ||
        strides[i] != expectedStride)
      break;
    ++rank;
    expectedStride *= shape[i];
  }
  return rank;
}

FailureOr<AffineExpr> linearizeToElementOffset(AffineMap map, MemRefType type) {
  if (map.getNumResults() != type.getShape().size())
    return failure();
  MLIRContext *ctx = type.getContext();
  ArrayRef<int64_t> shape = type.getShape();

  AffineMap layout;
  if (type.getLayout().isIdentity()) {
    if (!type.hasStaticShape())
      return failure();
    layout = AffineMap::get(shape.size(), 0, linearizeIndices(ctx, shape), ctx);
  } else if (auto strided = dyn_cast<StridedLayoutAttr>(type.getLayout())) {
    AffineExpr linear = getAffineConstantExpr(0, ctx);
    for (auto [i, stride] : llvm::enumerate(strided.getStrides())) {
      if (ShapedType::isDynamic(stride))
        return failure();
      linear = linear + getAffineDimExpr(i, ctx) * stride;
    }
    layout = AffineMap::get(shape.size(), 0, linear, ctx);
  } else {
    return failure();
  }

  // Deliberately not simplified: simplification rewrites `x mod c` as
  // `x - (x floordiv c) * c`, which is the same value but splits one
  // dimension across two correlated terms, and getAffineUpperBound reasons
  // term by term.
  return layout.compose(map).getResult(0);
}

//===----------------------------------------------------------------------===//
// Analysis of a single affine expression over a box
//===----------------------------------------------------------------------===//

namespace {

/// A weighted sum of the dimensions plus a constant.
struct AffineSum {
  SmallVector<int64_t> coefficients;
  int64_t constant = 0;
};

/// `expr` in that form, or nullopt if it is not one: a floordiv, a mod, or a
/// product of two dimensions all defeat it.
std::optional<AffineSum> matchAffineSum(AffineExpr expr, unsigned numDims) {
  AffineSum sum;
  sum.coefficients.assign(numDims, 0);

  SmallVector<std::pair<AffineExpr, int64_t>> worklist{{expr, 1}};
  while (!worklist.empty()) {
    auto [current, scale] = worklist.pop_back_val();
    switch (current.getKind()) {
    case AffineExprKind::Add: {
      auto add = cast<AffineBinaryOpExpr>(current);
      worklist.push_back({add.getLHS(), scale});
      worklist.push_back({add.getRHS(), scale});
      break;
    }
    case AffineExprKind::Mul: {
      auto mul = cast<AffineBinaryOpExpr>(current);
      auto factor = dyn_cast<AffineConstantExpr>(mul.getRHS());
      if (!factor)
        return std::nullopt;
      worklist.push_back({mul.getLHS(), scale * factor.getValue()});
      break;
    }
    case AffineExprKind::DimId: {
      unsigned pos = cast<AffineDimExpr>(current).getPosition();
      if (pos >= numDims)
        return std::nullopt;
      sum.coefficients[pos] += scale;
      break;
    }
    case AffineExprKind::Constant:
      sum.constant += scale * cast<AffineConstantExpr>(current).getValue();
      break;
    default:
      return std::nullopt;
    }
  }
  return sum;
}

/// Range of `expr` over the box `[0, extents)`.
std::optional<std::pair<int64_t, int64_t>>
evaluateInterval(AffineExpr expr, ArrayRef<int64_t> extents) {
  auto operands = [&](AffineExpr e)
      -> std::optional<
          std::pair<std::pair<int64_t, int64_t>, std::pair<int64_t, int64_t>>> {
    auto binary = cast<AffineBinaryOpExpr>(e);
    auto lhs = evaluateInterval(binary.getLHS(), extents);
    auto rhs = evaluateInterval(binary.getRHS(), extents);
    if (!lhs || !rhs)
      return std::nullopt;
    return std::make_pair(*lhs, *rhs);
  };

  switch (expr.getKind()) {
  case AffineExprKind::Constant: {
    int64_t value = cast<AffineConstantExpr>(expr).getValue();
    return std::make_pair(value, value);
  }
  case AffineExprKind::DimId: {
    unsigned pos = cast<AffineDimExpr>(expr).getPosition();
    if (pos >= extents.size() || extents[pos] < 1)
      return std::nullopt;
    return std::make_pair(int64_t{0}, extents[pos] - 1);
  }
  case AffineExprKind::Add: {
    auto sides = operands(expr);
    if (!sides)
      return std::nullopt;
    auto [lhs, rhs] = *sides;
    return std::make_pair(lhs.first + rhs.first, lhs.second + rhs.second);
  }
  case AffineExprKind::Mul: {
    auto sides = operands(expr);
    if (!sides)
      return std::nullopt;
    auto [lhs, rhs] = *sides;
    // One side is always a constant, so taking all four products costs
    // nothing and copes with either side being it, or with it being negative.
    int64_t products[] = {lhs.first * rhs.first, lhs.first * rhs.second,
                          lhs.second * rhs.first, lhs.second * rhs.second};
    return std::make_pair(*llvm::min_element(products),
                          *llvm::max_element(products));
  }
  case AffineExprKind::FloorDiv:
  case AffineExprKind::CeilDiv: {
    auto sides = operands(expr);
    if (!sides)
      return std::nullopt;
    auto [lhs, divisor] = *sides;
    if (divisor.first != divisor.second || divisor.first <= 0)
      return std::nullopt;
    bool ceil = expr.getKind() == AffineExprKind::CeilDiv;
    auto divide = [&](int64_t value) {
      return ceil ? llvm::divideCeilSigned(value, divisor.first)
                  : llvm::divideFloorSigned(value, divisor.first);
    };
    return std::make_pair(divide(lhs.first), divide(lhs.second));
  }
  case AffineExprKind::Mod: {
    auto sides = operands(expr);
    if (!sides)
      return std::nullopt;
    auto [lhs, modulus] = *sides;
    if (modulus.first != modulus.second || modulus.first <= 0)
      return std::nullopt;
    // Tight when the operand cannot wrap, the whole residue class otherwise.
    if (lhs.first >= 0 && lhs.second - lhs.first < modulus.first &&
        lhs.first % modulus.first <= lhs.second % modulus.first)
      return std::make_pair(lhs.first % modulus.first,
                            lhs.second % modulus.first);
    return std::make_pair(int64_t{0}, modulus.first - 1);
  }
  default:
    return std::nullopt;
  }
}

/// Whether no dimension appears twice in `expr`. Interval arithmetic treats
/// every occurrence as independent, so it is exact under this condition and
/// merely an over-approximation without it -- and an over-approximated bound
/// cannot be used to reject anything.
bool dimensionsOccurOnce(AffineExpr expr, unsigned numDims) {
  SmallVector<unsigned> counts(numDims, 0);
  bool ok = true;
  expr.walk([&](AffineExpr sub) {
    if (auto dim = dyn_cast<AffineDimExpr>(sub))
      if (dim.getPosition() < numDims && ++counts[dim.getPosition()] > 1)
        ok = false;
  });
  return ok;
}

} // namespace

std::optional<int64_t> getAffineUpperBound(AffineExpr expr,
                                           ArrayRef<int64_t> extents) {
  if (!dimensionsOccurOnce(expr, extents.size()))
    return std::nullopt;
  if (auto interval = evaluateInterval(expr, extents))
    return interval->second;
  return std::nullopt;
}

std::optional<bool> isAffineExprInjective(AffineExpr expr,
                                          ArrayRef<int64_t> extents) {
  std::optional<AffineSum> sum = matchAffineSum(expr, extents.size());
  if (!sum)
    return std::nullopt;

  // A dimension the expression ignores maps its whole range to one place, so
  // unless it is a single point it collides outright. This is the one case
  // that can be decided negatively, and it is the one that matters: it is what
  // a broadcast looks like. A negative coefficient only mirrors a range, so
  // magnitudes are what count below.
  SmallVector<std::pair<int64_t, int64_t>> terms;
  for (auto [coefficient, extent] : llvm::zip(sum->coefficients, extents)) {
    if (extent <= 1)
      continue;
    if (coefficient == 0)
      return false;
    terms.push_back({std::abs(coefficient), extent});
  }
  llvm::sort(terms);

  // Read the terms as digits of a mixed-radix number: if each one's stride
  // clears everything the smaller ones can reach, no two points coincide. The
  // converse does not hold -- {0,2,4} + {0,3} is injective and fails this --
  // so a failure means "cannot tell", not "collides".
  int64_t reach = 0;
  for (auto [stride, extent] : terms) {
    if (stride <= reach)
      return std::nullopt;
    reach += stride * (extent - 1);
  }
  return true;
}

/// The terms of a sum, as a flat list. `a + (b + c)` and `(a + b) + c` give
/// the same one, which is what lets the rules below reason term by term.
static void flattenAffineSum(AffineExpr expr,
                             SmallVectorImpl<AffineExpr> &terms) {
  if (auto add = llvm::dyn_cast<AffineBinaryOpExpr>(expr);
      add && add.getKind() == AffineExprKind::Add) {
    flattenAffineSum(add.getLHS(), terms);
    flattenAffineSum(add.getRHS(), terms);
    return;
  }
  terms.push_back(expr);
}

/// The constant a term is a multiple of, or nullopt when that is not
/// something this can establish.
static std::optional<int64_t> affineTermCoefficient(AffineExpr expr) {
  if (llvm::isa<AffineDimExpr>(expr) || llvm::isa<AffineSymbolExpr>(expr))
    return 1;
  if (auto constant = llvm::dyn_cast<AffineConstantExpr>(expr))
    return constant.getValue();
  if (auto mul = llvm::dyn_cast<AffineBinaryOpExpr>(expr);
      mul && mul.getKind() == AffineExprKind::Mul)
    if (auto c = llvm::dyn_cast<AffineConstantExpr>(mul.getRHS()))
      return c.getValue();
  return std::nullopt;
}

/// A term of a sum, split into its base expression and constant coefficient:
/// `x * c` gives (x, c), anything else is its own base with coefficient 1.
static std::pair<AffineExpr, int64_t> affineTermBase(AffineExpr term) {
  if (auto mul = llvm::dyn_cast<AffineBinaryOpExpr>(term);
      mul && mul.getKind() == AffineExprKind::Mul)
    if (auto c = llvm::dyn_cast<AffineConstantExpr>(mul.getRHS()))
      return {mul.getLHS(), c.getValue()};
  return {term, 1};
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
  case AffineExprKind::Mul: {
    AffineBinaryOpExpr binaryExpr = cast<AffineBinaryOpExpr>(expr);
    return getAffineBinaryOpExpr(
        expr.getKind(),
        simplifyAffineExprWithBounds(binaryExpr.getLHS(), numDims, numSymbols,
                                     dimLowerBounds, dimUpperBounds),
        simplifyAffineExprWithBounds(binaryExpr.getRHS(), numDims, numSymbols,
                                     dimLowerBounds, dimUpperBounds));
  }
  case AffineExprKind::Add: {
    SmallVector<AffineExpr> terms;
    flattenAffineSum(expr, terms);
    for (AffineExpr &term : terms)
      term = simplifyAffineExprWithBounds(term, numDims, numSymbols,
                                          dimLowerBounds, dimUpperBounds);

    // Telescope the pair of terms a subtracted floor division leaves behind:
    //
    //   c*x + (-c*k)*(x floordiv k)  ==  c*(x mod k)
    //
    // by the definition x mod k = x - k*(x floordiv k). When the base is
    // itself a division, its dividing pair shows up pre-merged (the floordiv
    // rule below turns (x floordiv a) floordiv k into x floordiv (a*k)), so
    // the same telescoping is recognized through that form:
    //
    //   c*(x floordiv a) + (-c*k)*(x floordiv (a*k))  ==  c*((x floordiv a) mod
    //   k)
    //
    // Delinearizing an index with respect to one tiling and relinearizing it
    // with respect to another (cnm-to-upmem block derivation does this)
    // produces sums like
    //   d1 + 16*(d1 floordiv 128) - 16*(d1 floordiv 16) - ...
    // whose pairs telescope into single mod terms. Each rewrite removes one
    // term from the sum, so the loop terminates.
    bool changed = true;
    while (changed) {
      changed = false;
      for (size_t i = 0; i < terms.size() && !changed; ++i) {
        auto [base, coefficient] = affineTermBase(terms[i]);
        int64_t baseDivisor = 1;
        AffineExpr baseOperand = base;
        if (auto baseDiv = llvm::dyn_cast<AffineBinaryOpExpr>(base);
            baseDiv && baseDiv.getKind() == AffineExprKind::FloorDiv)
          if (auto a = llvm::dyn_cast<AffineConstantExpr>(baseDiv.getRHS());
              a && a.getValue() > 0) {
            baseDivisor = a.getValue();
            baseOperand = baseDiv.getLHS();
          }
        for (size_t j = 0; j < terms.size() && !changed; ++j) {
          if (i == j)
            continue;
          auto [divTerm, divCoefficient] = affineTermBase(terms[j]);
          auto div = llvm::dyn_cast<AffineBinaryOpExpr>(divTerm);
          if (!div || div.getKind() != AffineExprKind::FloorDiv ||
              div.getLHS() != baseOperand)
            continue;
          auto m = llvm::dyn_cast<AffineConstantExpr>(div.getRHS());
          if (!m || m.getValue() <= 0 || m.getValue() % baseDivisor != 0)
            continue;
          int64_t k = m.getValue() / baseDivisor;
          if (k <= 1 || divCoefficient != -coefficient * k)
            continue;
          terms[i] = simplifyAffineExprWithBounds(
              (base % k) * coefficient, numDims, numSymbols, dimLowerBounds,
              dimUpperBounds);
          terms.erase(terms.begin() + j);
          changed = true;
        }
      }
    }

    AffineExpr sum = terms.front();
    for (AffineExpr term : llvm::drop_begin(terms))
      sum = sum + term;
    return sum;
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

      // Merge nested divisions:  (x floordiv a) floordiv b == x floordiv (a*b)
      // for positive constants a and b.
      if (kind == AffineExprKind::FloorDiv && rhs > 0) {
        if (auto inner = llvm::dyn_cast<AffineBinaryOpExpr>(sLHS);
            inner && inner.getKind() == AffineExprKind::FloorDiv)
          if (auto innerRhs =
                  llvm::dyn_cast<AffineConstantExpr>(inner.getRHS());
              innerRhs && innerRhs.getValue() > 0)
            return simplifyAffineExprWithBounds(
                inner.getLHS().floorDiv(innerRhs.getValue() * rhs), numDims,
                numSymbols, dimLowerBounds, dimUpperBounds);
      }
      if (lhsUB) {
        if (kind == AffineExprKind::Mod && *lhsUB < rhs) {
          return sLHS;
        } else if (kind == AffineExprKind::FloorDiv && *lhsUB < rhs) {
          return getAffineConstantExpr(0, expr.getContext());
        }
      }

      // Drop the terms of a remainder that the modulus divides: they
      // contribute nothing to it.
      //
      //   (A + B) mod m  ==  B mod m       when m divides every term of A
      //
      // Splitting a host dimension to line a transfer's blocks up with it
      // (cnm::alignHostToBlocks) produces exactly this shape -- the split of
      // `dpu*32 + tasklet*4 + i` at 4 leaves `(dpu*32 + tasklet*4 + i) mod 4`,
      // which is `i`. The recursive call is what then reaches that last step,
      // through the bound rule above; it terminates because the expression it
      // is given has strictly fewer terms.
      if (kind == AffineExprKind::Mod && rhs > 1) {
        SmallVector<AffineExpr> terms;
        flattenAffineSum(sLHS, terms);
        AffineExpr rest;
        bool dropped = false;
        for (AffineExpr term : terms) {
          std::optional<int64_t> coeff = affineTermCoefficient(term);
          if (coeff && *coeff % rhs == 0) {
            dropped = true;
            continue;
          }
          rest = rest ? rest + term : term;
        }
        if (dropped) {
          if (!rest)
            return getAffineConstantExpr(0, expr.getContext());
          return simplifyAffineExprWithBounds(rest % rhs, numDims, numSymbols,
                                              dimLowerBounds, dimUpperBounds);
        }
      }

      // Drop the low-order terms of a division that cannot influence the
      // quotient. If the dividend splits as A + B where A is a multiple of
      // some `g` dividing `rhs` and B is always smaller than `g`, then B can
      // never carry into the quotient:
      //
      //   (A + B) floordiv rhs  ==  (A floordiv g) floordiv (rhs / g)
      //
      // Proof: write A = g*a and rhs = g*q, and a = k*q + r with 0 <= r < q.
      // Then A + B = g*q*k + (g*r + B), and 0 <= g*r + B <= g*(q-1) + g-1
      // < g*q, so the quotient is exactly k.
      //
      // This is what turns a linearized workgroup index back into a
      // coordinate of a single workgroup dimension -- e.g. with 2048 DPUs of
      // 8 tasklets, (dpu*8 + tasklet) floordiv 512 is just dpu floordiv 64.
      // Downstream passes test whether a scatter map depends on a workgroup
      // dimension to decide whether a buffer is shared or replicated, and
      // that test is syntactic, so leaving `tasklet` in the expression costs
      // a real buffer.
      if (kind == AffineExprKind::FloorDiv && rhs > 1) {
        SmallVector<AffineExpr> terms;
        flattenAffineSum(sLHS, terms);

        for (int64_t g = rhs; g > 1; --g) {
          if (rhs % g != 0)
            continue;
          AffineExpr big;
          int64_t restUB = 0;
          bool usable = true;
          for (AffineExpr term : terms) {
            std::optional<int64_t> coeff = affineTermCoefficient(term);
            if (coeff && *coeff % g == 0) {
              big = big ? big + term : term;
              continue;
            }
            auto ub =
                getBoundForAffineExpr(term, numDims, numSymbols, dimLowerBounds,
                                      dimUpperBounds, true);
            if (!ub || *ub < 0) {
              usable = false;
              break;
            }
            restUB += *ub;
          }
          if (!usable || !big || restUB >= g)
            continue;
          return big.floorDiv(g).floorDiv(rhs / g);
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
    // The upstream simplifier flattens `(x mod k) * c` into its defining
    // floordiv pair and cannot reconstruct the mod under the coefficient,
    // undoing the telescoping done above. Run it for its own
    // canonicalizations, then telescope once more so the mod form is what
    // survives.
    e = simplifyAffineExpr(e, map.getNumDims(), map.getNumSymbols());
    e = simplifyAffineExprWithBounds(e, map.getNumDims(), map.getNumSymbols(),
                                     lowerBounds, upperBounds);
    exprs.push_back(e);
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
  if (auto memrefTy = dyn_cast<MemRefType>(type)) {
    // Use identity (null) layout for the target type: cloneWith would preserve
    // any strided layout from the source, which causes a rank mismatch when the
    // new shape has a different rank than the strides count.
    auto newTy =
        MemRefType::get(newShape, memrefTy.getElementType(),
                        MemRefLayoutAttrInterface{}, memrefTy.getMemorySpace());
    auto shapeBuf = memref::AllocaOp::create(
        builder, loc, MemRefType::get({newTy.getRank()}, builder.getI64Type()));
    for (auto [i, dim] : llvm::enumerate(newShape)) {
      auto idx = arith::ConstantIndexOp::create(builder, loc, i);
      auto dimSize = arith::ConstantOp::create(builder, loc,
                                               builder.getI64IntegerAttr(dim));
      memref::StoreOp::create(builder, loc, dimSize, shapeBuf, ValueRange{idx});
    }
    return dyn_cast<TypedValue<ShapedType>>(
        memref::ReshapeOp::create(builder, loc, newTy, value, shapeBuf)
            .getResult());
  }

  auto newTy = type.cloneWith(newShape, type.getElementType());
  assert(isa<RankedTensorType>(newTy) && "must be memref or tensor");
  auto reifiedShape = arith::ConstantOp::create(
      builder, loc,
      RankedTensorType::get({newTy.getRank()}, builder.getI64Type()),
      builder.getI64TensorAttr(newShape));
  return dyn_cast<TypedValue<ShapedType>>(
      tensor::ReshapeOp::create(builder, loc, newTy, value, reifiedShape)
          .getResult());
}

//===--------------------------------------------------------------------===//
// Tiling into affine.for loops
//===--------------------------------------------------------------------===//

/// `min(tileSize, ub - offset)`: the last tile is short when the tile size does
/// not divide the loop range. Mirrors what `scf::tileUsingSCF` computes for its
/// own loops -- the tiled op is built from these, so it has to agree.
static OpFoldResult boundedTileSize(OpBuilder &b, Location loc, Range range,
                                    OpFoldResult offset,
                                    OpFoldResult tileSize) {
  std::optional<int64_t> size = getConstantIntValue(tileSize);
  if (size && *size == 1)
    return tileSize;

  std::optional<int64_t> lb = getConstantIntValue(range.offset);
  std::optional<int64_t> ub = getConstantIntValue(range.size);
  if (lb && ub && size && (*ub - *lb) % *size == 0)
    return tileSize;

  AffineExpr d0, s0, s1;
  bindDims(b.getContext(), d0);
  bindSymbols(b.getContext(), s0, s1);
  AffineMap minMap = AffineMap::get(1, 2, {s0 - d0, s1}, b.getContext());
  return affine::makeComposedFoldedAffineMin(
      b, loc, minMap, SmallVector<OpFoldResult>{offset, range.size, tileSize});
}

/// Build the inter-tile loops as an `affine.for` nest and leave the rewriter
/// pointing inside the innermost one, where the tiled body goes.
static FailureOr<scf::SCFTilingOptions::CustomLoopHeaderInfo>
generateAffineTileLoops(RewriterBase &rewriter, Location loc,
                        ArrayRef<Range> loopRanges,
                        ArrayRef<OpFoldResult> tileSizes,
                        ValueRange destinationTensors) {
  SmallVector<LoopLikeOpInterface> loops;
  SmallVector<Value> ivs;
  for (auto [range, tileSize] : llvm::zip_equal(loopRanges, tileSizes)) {
    if (isZeroInteger(tileSize))
      continue; // dimension not tiled, so no loop over it

    // An affine.for takes its bounds as affine maps over valid dims and
    // symbols. Constants are the only form guaranteed to be valid wherever
    // this runs, hence the check in `canTileUsingAffineFor`.
    std::optional<int64_t> lb = getConstantIntValue(range.offset);
    std::optional<int64_t> ub = getConstantIntValue(range.size);
    std::optional<int64_t> step = getConstantIntValue(tileSize);
    if (!lb || !ub || !step)
      return rewriter.notifyMatchFailure(
          loc,
          "cannot tile a dimension of non-constant bounds into affine.for");

    auto loop = affine::AffineForOp::create(rewriter, loc, *lb, *ub, *step);
    loops.push_back(loop);
    ivs.push_back(loop.getInductionVar());
    rewriter.setInsertionPoint(loop.getBody()->getTerminator());
  }

  // The tile the innermost body covers, in iteration-space coordinates.
  SmallVector<OpFoldResult> offsets, sizes;
  unsigned ivIdx = 0;
  for (auto [range, tileSize] : llvm::zip_equal(loopRanges, tileSizes)) {
    if (isZeroInteger(tileSize)) {
      offsets.push_back(range.offset);
      sizes.push_back(range.size);
      continue;
    }
    OpFoldResult offset = getAsOpFoldResult(ivs[ivIdx++]);
    offsets.push_back(offset);
    sizes.push_back(boundedTileSize(rewriter, loc, range, offset, tileSize));
  }

  return scf::SCFTilingOptions::CustomLoopHeaderInfo{
      loops, offsets, sizes, llvm::to_vector(destinationTensors)};
}

/// Terminate the loops built by `generateAffineTileLoops`. Nothing to do for
/// the ops we accept: they have pure buffer semantics, so no tile is yielded
/// back, and an `affine.for` without iter_args is created already terminated.
static LogicalResult finishAffineTileLoops(RewriterBase &, Location loc,
                                           ArrayRef<LoopLikeOpInterface>,
                                           ValueRange tiledResults,
                                           ArrayRef<SmallVector<OpFoldResult>>,
                                           ArrayRef<SmallVector<OpFoldResult>>,
                                           ValueRange) {
  if (!tiledResults.empty())
    return emitError(loc) << "cannot tile an op with " << tiledResults.size()
                          << " results into affine.for loops: only buffer "
                             "semantics are supported";
  return success();
}

bool canTileUsingAffineFor(TilingInterface op, ArrayRef<int64_t> tileSizes) {
  if (op->getNumResults() != 0)
    return false;

  // The iteration domain is only available as IR, which we are in no position
  // to build here; take the static ranges off the op instead.
  auto indexed = dyn_cast<IndexingMapOpInterface>(op.getOperation());
  if (!indexed)
    return false;

  SmallVector<int64_t> ranges = indexed.getStaticLoopRanges();
  if (ranges.size() != tileSizes.size())
    return false;
  for (auto [range, tileSize] : llvm::zip_equal(ranges, tileSizes))
    if (tileSize != 0 && ShapedType::isDynamic(range))
      return false;
  return true;
}

FailureOr<scf::SCFTilingResult>
tileUsingAffineFor(RewriterBase &rewriter, TilingInterface op,
                   scf::SCFTilingOptions options, ArrayRef<int64_t> tileSizes) {
  if (!canTileUsingAffineFor(op, tileSizes))
    return rewriter.notifyMatchFailure(op,
                                       "cannot be tiled into affine.for loops");

  options.setTileSizes(getAsIndexOpFoldResult(rewriter.getContext(), tileSizes))
      .setLoopType(scf::SCFTilingOptions::LoopType::CustomOp)
      .setCustomLoopGenerationFns(generateAffineTileLoops,
                                  finishAffineTileLoops);
  return scf::tileUsingSCF(rewriter, op, options);
}

} // namespace mlir
