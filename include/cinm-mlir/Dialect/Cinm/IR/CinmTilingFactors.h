#pragma once

#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Transform/Utils/DiagnosedSilenceableFailure.h>

namespace mlir {
class Operation;
namespace cinm {

/// Compute tiling factors for a cinm tiling operation using hardware
/// parameters. Dispatches on the op type to produce the appropriate number of
/// factors (e.g. 1 for elementwise, 3 for gemm). The factors are appended to
/// tilingFactors. Returns silenceableFailure if tile sizes cannot be determined
/// automatically and must be provided via cinm.tile_sizes.
DiagnosedSilenceableFailure
computeTilingFactorsForOp(int64_t leafBufferSizeBytes,
                          ArrayRef<int64_t> workgroupShape, Operation *op,
                          SmallVectorImpl<int64_t> &tilingFactors);

} // namespace cinm
} // namespace mlir
