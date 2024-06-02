

#include <mlir/Pass/Pass.h>

namespace mlir::cinm {

    /// Full pipeline
    void registerCinmToCnmPipeline();

    /// Just the pass after --cinm-tiling
    std::unique_ptr<Pass> createConvertTiledCinmToCnmPass();
} // namespace mlir::cnm