

#include <cinm-mlir/Dialect/Cim/IR/CimDialect.h>
#include <mlir/Pass/Pass.h>

namespace mlir::cinm {

/// Full pipeline
void registerCinmToCimPipeline();

/// Just the pass after --cinm-tiling
std::unique_ptr<Pass> createConvertTiledCinmToCimPass();
} // namespace mlir::cinm