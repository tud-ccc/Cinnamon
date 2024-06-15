#pragma once

#include <mlir/Pass/Pass.h>
#include <mlir/Conversion/Passes.h>

#ifdef CINM_GPU_SUPPORT
namespace mlir::cnm {
    void populateCnmToGPUFinalTypeConversions(LLVMTypeConverter &typeConverter);
    void populateCnmToGPUConversionPatterns(LLVMTypeConverter &typeConverter, RewritePatternSet &patterns);
    std::unique_ptr<Pass> createConvertCnmToGPUPass();
} // namespace mlir::cnm

#endif