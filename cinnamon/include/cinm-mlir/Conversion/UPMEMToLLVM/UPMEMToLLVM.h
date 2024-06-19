#pragma once

#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMDialect.h"

#include <mlir/Conversion/Passes.h>
#include <mlir/Pass/Pass.h>

namespace mlir::upmem {

void populateUPMEMToLLVMFinalTypeConversions(LLVMTypeConverter &typeConverter);

void populateUPMEMToLLVMConversionPatterns(LLVMTypeConverter &typeConverter,
                                           RewritePatternSet &patterns);

std::unique_ptr<Pass> createConvertUPMEMToLLVMPass();

void registerConvertUpmemToLLvmInterface(DialectRegistry &registry);
} // namespace mlir::upmem
