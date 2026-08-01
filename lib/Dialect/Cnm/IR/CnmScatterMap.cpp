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
  // dimension across two correlated terms, and the analyses below reason
  // term by term.
  return layout.compose(map).getResult(0);
}

//===----------------------------------------------------------------------===//
// Analysis
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
      -> std::optional<std::pair<std::pair<int64_t, int64_t>,
                                 std::pair<int64_t, int64_t>>> {
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

} // namespace

/// Whether no dimension appears twice in `expr`. Interval arithmetic treats
/// every occurrence as independent, so it is exact under this condition and
/// merely an over-approximation without it -- and an over-approximated bound
/// cannot be used to reject anything.
static bool dimensionsOccurOnce(AffineExpr expr, unsigned numDims) {
  SmallVector<unsigned> counts(numDims, 0);
  bool ok = true;
  expr.walk([&](AffineExpr sub) {
    if (auto dim = dyn_cast<AffineDimExpr>(sub))
      if (dim.getPosition() < numDims && ++counts[dim.getPosition()] > 1)
        ok = false;
  });
  return ok;
}

std::optional<int64_t>
mlir::cnm::getAffineUpperBound(AffineExpr expr, ArrayRef<int64_t> extents) {
  if (!dimensionsOccurOnce(expr, extents.size()))
    return std::nullopt;
  if (auto interval = evaluateInterval(expr, extents))
    return interval->second;
  return std::nullopt;
}

std::optional<bool>
mlir::cnm::isAffineExprInjective(AffineExpr expr, ArrayRef<int64_t> extents) {
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
