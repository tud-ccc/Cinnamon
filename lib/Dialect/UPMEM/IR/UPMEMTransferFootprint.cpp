//===- UPMEMTransferFootprint.cpp - Distinct host bytes of a transfer -----===//

#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMTransferFootprint.h"

#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMTypes.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <utility>
#include <vector>

using namespace mlir;

namespace {

/// `expr` at dimension value `d`, in plain integer arithmetic: no context
/// uniquing per evaluation, which matters at thousands of DPUs per transfer
/// and many transfers per search. None for a symbol.
std::optional<int64_t> evaluate(AffineExpr expr, int64_t d) {
  switch (expr.getKind()) {
  case AffineExprKind::Constant:
    return cast<AffineConstantExpr>(expr).getValue();
  case AffineExprKind::DimId:
    return d;
  case AffineExprKind::SymbolId:
    return std::nullopt;
  default:
    break;
  }
  auto binary = cast<AffineBinaryOpExpr>(expr);
  std::optional<int64_t> lhs = evaluate(binary.getLHS(), d);
  std::optional<int64_t> rhs = evaluate(binary.getRHS(), d);
  if (!lhs || !rhs)
    return std::nullopt;
  switch (expr.getKind()) {
  case AffineExprKind::Add:
    return *lhs + *rhs;
  case AffineExprKind::Mul:
    return *lhs * *rhs;
  case AffineExprKind::Mod:
    // Affine mod is non-negative, unlike C++'s %.
    return *rhs > 0 ? std::optional<int64_t>(((*lhs % *rhs) + *rhs) % *rhs)
                    : std::nullopt;
  case AffineExprKind::FloorDiv:
    return *rhs != 0
               ? std::optional<int64_t>(llvm::divideFloorSigned(*lhs, *rhs))
               : std::nullopt;
  case AffineExprKind::CeilDiv:
    return *rhs != 0
               ? std::optional<int64_t>(llvm::divideCeilSigned(*lhs, *rhs))
               : std::nullopt;
  default:
    return std::nullopt;
  }
}

} // namespace

std::optional<int64_t> upmem::uniqueHostElements(AffineMap map, int64_t numDpus,
                                                 ArrayRef<int64_t> strides,
                                                 int64_t count) {
  if (map.getNumDims() != 1 || map.getNumSymbols() != 0 ||
      map.getNumResults() != strides.size() || numDpus <= 0 || count <= 0)
    return std::nullopt;

  std::vector<int64_t> starts;
  starts.reserve(numDpus);
  for (int64_t dpu = 0; dpu < numDpus; ++dpu) {
    int64_t start = 0;
    for (auto [expr, stride] : llvm::zip_equal(map.getResults(), strides)) {
      std::optional<int64_t> index = evaluate(expr, dpu);
      if (!index)
        return std::nullopt;
      start += *index * stride;
    }
    starts.push_back(start);
  }

  // The union of [start, start + count) over all DPUs: sort the starts and
  // extend a running interval while the next one overlaps or touches it.
  std::sort(starts.begin(), starts.end());
  int64_t unique = 0;
  int64_t runStart = starts.front(), runEnd = starts.front() + count;
  for (int64_t start : llvm::drop_begin(starts)) {
    if (start > runEnd) {
      unique += runEnd - runStart;
      runStart = start;
    }
    runEnd = std::max(runEnd, start + count);
  }
  return unique + (runEnd - runStart);
}

std::optional<int64_t> upmem::uniqueHostBytes(ScatterOnArrayOp op) {
  auto type = cast<MemRefType>(op.getHostBuffer().getType());
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(type.getStridesAndOffset(strides, offset)) ||
      llvm::any_of(strides, ShapedType::isDynamic))
    return std::nullopt;
  auto hierarchy = cast<DeviceHierarchyType>(op.getHierarchy().getType());
  std::optional<int64_t> elements =
      uniqueHostElements(op.getScatterMap(), hierarchy.getNumDpus(), strides,
                         op.getTransferCount());
  if (!elements)
    return std::nullopt;
  return *elements * type.getElementTypeBitWidth() / 8;
}
