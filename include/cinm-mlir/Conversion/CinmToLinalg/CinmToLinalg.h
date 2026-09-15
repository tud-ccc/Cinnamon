#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir::cinm {

void populateCinmOpsToLinalgPatterns(RewritePatternSet &patterns,
                                     MLIRContext *ctx);
std::unique_ptr<Pass> createConvertCinmOpsToLinalgPass();

} // namespace mlir::cinm
