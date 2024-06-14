#pragma once

#include <mlir/Pass/Pass.h>
#include <mlir/Conversion/Passes.h>

namespace mlir::cnm {
    void populateCnmToGPUFinalTypeConversions(LLVMTypeConverter &typeConverter);
    void populateCnmToGPUConversionPatterns(LLVMTypeConverter &typeConverter, RewritePatternSet &patterns);
    std::unique_ptr<Pass> createConvertCnmToGPUPass();
} // namespace mlir::cnm
