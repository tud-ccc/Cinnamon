

#include "cinm-mlir/Dialect/Memristor/IR/MemristorDialect.h"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Pass/Pass.h>

namespace mlir::memristor {
void populateMemristorToFuncConversionPatterns(RewritePatternSet &patterns,
                                               MLIRContext *context);
std::unique_ptr<Pass> createConvertMemristorToFuncPass();
} // namespace mlir::memristor