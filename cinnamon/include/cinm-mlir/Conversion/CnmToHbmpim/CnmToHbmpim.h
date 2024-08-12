#pragma once

#include "cinm-mlir/Dialect/Hbmpim/IR/HbmpimDialect.h"

#include <mlir/Conversion/Passes.h>
#include <mlir/Pass/Pass.h>

namespace mlir::cnm {
void populateCnmToHbmpimFinalTypeConversions(TypeConverter &typeConverter);
void populateCnmToHbmpimConversionPatterns(TypeConverter &typeConverter,
                                          RewritePatternSet &patterns);
std::unique_ptr<Pass> createConvertCnmToHbmpimPass();
} // namespace mlir::cnm
