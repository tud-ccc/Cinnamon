#pragma once

#include <mlir/Conversion/Passes.h>
#include <mlir/Pass/Pass.h>

#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMDialect.h"

namespace mlir::cnm {
void populateCnmToUPMEMFinalTypeConversions(LLVMTypeConverter &typeConverter);
void populateCnmToUPMEMConversionPatterns(LLVMTypeConverter &typeConverter,
                                          RewritePatternSet &patterns);
std::unique_ptr<Pass> createConvertCnmToUPMEMPass();
} // namespace mlir::cnm
