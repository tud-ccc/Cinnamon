#include "cinm-mlir/Dialect/Cinm/IR/CinmTilingFactors.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <tuple>

using namespace mlir;
using namespace mlir::cinm;

namespace {

struct TilingParameters {
  /// Workgroup shape being considered for tiling.
  const ArrayRef<int64_t> workgroupShape;
  /// Sizes of the different buffer levels.
  const ArrayRef<int64_t> bufferSizesInBytes;

  // Opaque tiling sizes provided by the compute block (operation-agnostic).
  // Ordering/meaning is decided by each consuming op.
  std::optional<SmallVector<int64_t, 8>> tileSizes;

  TilingParameters(ArrayRef<int64_t> bufferSizesInBytes,
                   ArrayRef<int64_t> workgroupShape)
      : workgroupShape(workgroupShape), bufferSizesInBytes(bufferSizesInBytes) {
  }

  int64_t reduceClusterSize(int64_t numBuffers, int64_t reducedElements,
                            Type elementTy, int64_t extraElements = 0) const;
  std::optional<std::pair<int64_t, int64_t>>
  parallelClusterSize(int64_t n, int64_t m) const;
  int64_t workingGroupSize() const;
  int64_t bufferSizeOfLeaf() const;
  int64_t maxNumElementsOfType(Type ty) const;

  // helpers to access the opaque tile sizes.
  std::optional<ArrayRef<int64_t>> getTileSizes() const {
    if (!tileSizes)
      return std::nullopt;
    return ArrayRef<int64_t>(*tileSizes);
  }
  std::optional<int64_t> getTileSize(size_t i) const {
    if (!tileSizes || i >= tileSizes->size())
      return std::nullopt;
    return (*tileSizes)[i];
  }
  // explicit tiles if provided by compute.tileSizes
  // Convention for now: [parallel0, parallel1, reduction] for GEMM.
  std::optional<std::pair<int64_t, int64_t>> getProvidedParallelTiles() const {
    if (!tileSizes || tileSizes->size() < 2)
      return std::nullopt;
    return {{(*tileSizes)[0], (*tileSizes)[1]}};
  }
  std::optional<int64_t> getProvidedReductionTile() const {
    if (!tileSizes || tileSizes->size() < 3)
      return std::nullopt;
    return (*tileSizes)[2];
  }
};

/// Return the size of tiles on a reduce dimension.
/// Computes this by assuming the reduction operation needs (maybe several)
/// buffers of the same size, same element type. The returned tile size
/// divides the number of reduced elements.
/// This cannot fail but may return 1, meaning no blocking possible.
int64_t TilingParameters::reduceClusterSize(int64_t numBuffers,
                                            int64_t reducedElements,
                                            Type elementTy,
                                            int64_t extraElements) const {
  // in number of elements
  auto maxSizePerBuffer =
      (maxNumElementsOfType(elementTy) - extraElements) / numBuffers;
  // Now we need to find the largest divisor of `reducedElements` that is
  // smaller than maxSizePerBuffer
  for (int i = maxSizePerBuffer; i > 0; i--) {
    if (reducedElements % i == 0)
      return i;
  }
  return 1;
}

using OptionalTileSizes = std::optional<std::pair<int64_t, int64_t>>;

/// Determine tiling factors for dimensions n and m.
OptionalTileSizes TilingParameters::parallelClusterSize(int64_t n,
                                                        int64_t m) const {
  /// need to find a number that divides parallelElements and the working
  /// group size
  SmallVector<int64_t, 4> wgShape(workgroupShape);
  auto it = std::remove(wgShape.begin(), wgShape.end(), 1);
  if (it != wgShape.end())
    wgShape.erase(it);

  if (wgShape.size() == 2) {
    // try to fit perfectly
    if (n % wgShape[0] == 0 && m % wgShape[1] == 0)
      return OptionalTileSizes({wgShape[0], wgShape[1]});
    else if (n % wgShape[1] == 0 && m % wgShape[0] == 0)
      return OptionalTileSizes({wgShape[1], wgShape[0]});
  }

  auto wg = workingGroupSize();
  if (wg > m * n)
    return std::nullopt;
  auto a = std::gcd(n, wg);
  auto b = std::gcd(m, wg);

  if (a * b == wg)
    return OptionalTileSizes({a, b});
  else if (a > b)
    return OptionalTileSizes({a, wg / a});
  else if (b != 1)
    return OptionalTileSizes({wg / b, b});
  else
    return std::nullopt;
}

/// Number of parallel elements in the working group.
int64_t TilingParameters::workingGroupSize() const {
  return std::reduce(workgroupShape.begin(), workgroupShape.end(), int64_t{1},
                     std::multiplies<>());
}

int64_t TilingParameters::bufferSizeOfLeaf() const {
  // Buffers at one level are shared with later levels.
  // For a workgroup {A,B,C} and buffer sizes {M,N,P},
  // per-leaf space is P + N/C + M/B/C.
  int64_t numLeafsInDim = 1;
  int64_t bufSize = 0;
  int i = 0;
  do {
    size_t lastIdx = bufferSizesInBytes.size() - 1 - i;
    bufSize += bufferSizesInBytes[lastIdx] / numLeafsInDim;
    numLeafsInDim *=
        lastIdx < workgroupShape.size() ? workgroupShape[lastIdx] : 1;
    i++;
  } while (i < static_cast<int64_t>(bufferSizesInBytes.size()));
  return bufSize;
}

int64_t TilingParameters::maxNumElementsOfType(Type ty) const {
  int64_t bw =
      std::max<int64_t>(8, static_cast<int64_t>(ty.getIntOrFloatBitWidth()));
  return bufferSizeOfLeaf() / (bw / 8);
}

} // namespace

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
           << params.workgroupShape << "; provide cinm.tile_sizes [tM,tN,tK]";
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

DiagnosedSilenceableFailure mlir::cinm::computeTilingFactorsForOp(
    int64_t leafBufferSizeBytes, ArrayRef<int64_t> workgroupShape,
    Operation *op, SmallVectorImpl<int64_t> &tilingFactors) {
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
