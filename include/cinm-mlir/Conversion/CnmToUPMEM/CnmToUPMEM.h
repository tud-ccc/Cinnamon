#pragma once

#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMDialect.h"

#include <mlir/Conversion/Passes.h>
#include <mlir/Pass/Pass.h>

namespace mlir::cnm {
void populateCnmToUPMEMFinalTypeConversions(TypeConverter &typeConverter);
void populateCnmToUPMEMConversionPatterns(TypeConverter &typeConverter,
                                          RewritePatternSet &patterns);
std::unique_ptr<Pass> createConvertCnmToUPMEMPass();
} // namespace mlir::cnm
