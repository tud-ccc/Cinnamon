#include <mlir/Pass/Pass.h>

namespace mlir {

// The generated pass base for ConvertTiledCinmToCnm lives in `mlir`, so its
// options struct has to as well.
#define GEN_PASS_DECL_CONVERTTILEDCINMTOCNM
#include "cinm-mlir/Conversion/CinmPasses.h.inc"

namespace cinm {

/// Full pipeline
void registerCinmToCnmPipeline();

/// Just the pass after --cinm-tiling
std::unique_ptr<Pass> createConvertTiledCinmToCnmPass();
std::unique_ptr<Pass>
    createConvertTiledCinmToCnmPass(ConvertTiledCinmToCnmOptions);
} // namespace cinm
} // namespace mlir
