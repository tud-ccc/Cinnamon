#include "cinm-mlir/Dialect/Cinm/IR/CinmTilingFactors.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/IR/TilingParameters.h"

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <tuple>

using namespace mlir;
using namespace mlir::cinm;

// ---------------------------------------------------------------------------
// Per-op helpers — each computes tiling factors from hardware params.
// ---------------------------------------------------------------------------

static FailureOr<int64_t> getReduceTiles(ReduceOp op,
                                         const TilingParameters &params) {
  auto ty = op.getInput().getType();
  return params.reduceClusterSize(1, ty.getNumElements(), ty.getElementType());
}

static FailureOr<int64_t> getElementwiseTiles(ElementwiseOp op,
                                              const TilingParameters &params) {
  // No automatic heuristic for elementwise ops yet — require explicit sizes.
  (void)params;
  return op->emitError(
      "cannot determine tiling factors automatically for elementwise op; "
      "set cinm.tile_sizes explicitly");
}

static FailureOr<std::tuple<int64_t, int64_t, int64_t>>
getGemmTiles(const int64_t M, const int64_t N, const int64_t K,
             const Type eltType, const TilingParameters &params,
             Operation *errorLoc) {
  auto parallelTileSizes = params.parallelClusterSize(M, N);
  if (!parallelTileSizes)
    return errorLoc->emitError("cannot determine tiling factors for M=")
           << M << ", N=" << N << " and working group shape "
           << params.workgroupShape
           << "; provide cinm.tile_sizes [tM,tN,tK]";
  auto [p0, p1] = *parallelTileSizes;
  int64_t r = params.reduceClusterSize(2, K, eltType, /*extraElements=*/1);
  return std::make_tuple(p0, p1, r);
}

static FailureOr<std::tuple<int64_t, int64_t>>
getGemvTiles(const int64_t M, const int64_t K, const Type eltType,
             const TilingParameters &params, Operation *errorLoc) {
  if (ShapedType::isDynamic(M) || ShapedType::isDynamic(K))
    return errorLoc->emitError(
        "cannot determine tiling factors for dynamic dimensions; "
        "provide cinm.tile_sizes [tM,tK]");

  auto parallelTileSize = params.parallelClusterSize(M, 1);
  if (!parallelTileSize)
    return errorLoc->emitError("cannot determine tiling factors for M=")
           << M << " and working group shape " << params.workgroupShape
           << "; provide cinm.tile_sizes [tM,tK]";
  auto [p, _] = *parallelTileSize;
  int64_t k = params.reduceClusterSize(2, K, eltType, /*extraElements=*/1);
  return std::make_tuple(p, k);
}

static FailureOr<int64_t> getActivateTiles(ActivateOp op,
                                           const TilingParameters &params) {
  auto inTy = op.getInput().getType();
  const int64_t total = inTy.getNumElements();
  int64_t p = 0;
  if (auto par = params.parallelClusterSize(total, 1))
    p = std::max<int64_t>(1, par->first);
  if (p <= 0)
    p = std::max<int64_t>(1, params.workingGroupSize());
  p = std::min<int64_t>(p, total);
  if (p <= 0)
    return op->emitError("cannot determine tiling factor for activate op");
  return p;
}

// ---------------------------------------------------------------------------
// Public dispatcher
// ---------------------------------------------------------------------------

DiagnosedSilenceableFailure
mlir::cinm::computeTilingFactorsForOp(int64_t leafBufferSizeBytes,
                                      ArrayRef<int64_t> workgroupShape,
                                      Operation *op,
                                      SmallVectorImpl<int64_t> &tilingFactors) {
  // Build TilingParameters from hardware descriptors. Using a single-level
  // buffer array of size leafBufferSizeBytes so that bufferSizeOfLeaf()
  // returns that value directly, while workgroupShape drives parallelism.
  SmallVector<int64_t, 1> bufSizes = {leafBufferSizeBytes};
  TilingParameters params(bufSizes, workgroupShape);

  if (auto reduceOp = dyn_cast<ReduceOp>(op)) {
    auto factor = getReduceTiles(reduceOp, params);
    if (failed(factor))
      return DiagnosedSilenceableFailure::definiteFailure();
    tilingFactors.push_back(*factor);
    return DiagnosedSilenceableFailure::success();
  }

  if (auto elemwiseOp = dyn_cast<ElementwiseOp>(op)) {
    auto factor = getElementwiseTiles(elemwiseOp, params);
    if (failed(factor))
      return DiagnosedSilenceableFailure::definiteFailure();
    tilingFactors.push_back(*factor);
    return DiagnosedSilenceableFailure::success();
  }

  if (auto gemmOp = dyn_cast<GemmOp>(op)) {
    auto lhsTy = gemmOp.getLhs().getType();
    auto rhsTy = gemmOp.getRhs().getType();
    if (lhsTy.getRank() != 2 || rhsTy.getRank() != 2)
      return DiagnosedSilenceableFailure::definiteFailure();
    const int64_t M = lhsTy.getDimSize(0);
    const int64_t K = lhsTy.getDimSize(1);
    const int64_t N = rhsTy.getDimSize(1);
    if (ShapedType::isDynamic(M) || ShapedType::isDynamic(K) ||
        ShapedType::isDynamic(N))
      return DiagnosedSilenceableFailure::definiteFailure();
    auto tiles = getGemmTiles(M, N, K, lhsTy.getElementType(), params, op);
    if (failed(tiles))
      return DiagnosedSilenceableFailure::definiteFailure();
    auto [p0, p1, r] = *tiles;
    tilingFactors.append({p0, p1, r});
    return DiagnosedSilenceableFailure::success();
  }

  if (auto gemvOp = dyn_cast<GemvOp>(op)) {
    auto aTy = cast<ShapedType>(gemvOp.getLhs().getType());
    const int64_t M = aTy.getDimSize(0);
    const int64_t K = aTy.getDimSize(1);
    auto tiles = getGemvTiles(M, K, aTy.getElementType(), params, op);
    if (failed(tiles))
      return DiagnosedSilenceableFailure::definiteFailure();
    auto [pM, rK] = *tiles;
    tilingFactors.append({pM, rK});
    return DiagnosedSilenceableFailure::success();
  }

  if (auto activateOp = dyn_cast<ActivateOp>(op)) {
    auto factor = getActivateTiles(activateOp, params);
    if (failed(factor))
      return DiagnosedSilenceableFailure::definiteFailure();
    tilingFactors.push_back(*factor);
    return DiagnosedSilenceableFailure::success();
  }

  // BatchGemmOp and BatchGemvOp have no automatic heuristic.
  if (isa<BatchGemmOp>(op))
    return emitSilenceableFailure(op->getLoc())
           << "cannot determine tiling factors automatically for batch_gemm; "
              "set cinm.tile_sizes [batch,tM,tN,tK] explicitly";
  if (isa<BatchGemvOp>(op))
    return emitSilenceableFailure(op->getLoc())
           << "cannot determine tiling factors automatically for batch_gemv; "
              "set cinm.tile_sizes [batch,tM,tK] explicitly";

  return emitSilenceableFailure(op->getLoc())
         << "unrecognized cinm op for tiling factor computation: " << *op;
}
