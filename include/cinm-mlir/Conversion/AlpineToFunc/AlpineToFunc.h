

#include "cinm-mlir/Dialect/Alpine/IR/AlpineDialect.h"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Pass/Pass.h>

namespace mlir::alpine {
void populateAlpineToFuncConversionPatterns(RewritePatternSet &patterns,
                                            MLIRContext *context);
std::unique_ptr<Pass> createConvertAlpineToFuncPass();
} // namespace mlir::alpine
