#pragma once

#include <mlir/Pass/Pass.h>

#include <memory>

namespace mlir {
class MLIRContext;
class RewritePatternSet;
class Pass;

namespace cinm {

std::unique_ptr<Pass> createIm2ColToMatmulPass();
void populateIm2ColToMatmulPatterns(RewritePatternSet &patterns,
                                    MLIRContext *context);

} // namespace cinm
} // namespace mlir
