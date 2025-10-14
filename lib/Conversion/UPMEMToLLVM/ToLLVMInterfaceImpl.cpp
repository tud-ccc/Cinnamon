#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMBase.h"
#include <cinm-mlir/Conversion/UPMEMToLLVM/UPMEMToLLVM.h>
#include <mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h>
#include <mlir/IR/DialectRegistry.h>

namespace mlir::upmem {

struct UpmemToLlvmDialectInterface : public ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;

  /// Hook for derived dialect interface to load the dialects they
  /// target. The LLVMDialect is implicitly already loaded, but this
  /// method allows to load other intermediate dialects used in the
  /// conversion, or target dialects like NVVM for example.
  void loadDependentDialects(MLIRContext *) const override {}

  /// Hook for derived dialect interface to provide conversion patterns
  /// and mark dialect legal for the conversion target.
  void populateConvertToLLVMConversionPatterns(
      ConversionTarget &target, LLVMTypeConverter &typeConverter,
      RewritePatternSet &patterns) const override {
    target.addIllegalDialect<upmem::UPMEMDialect>();
    populateUPMEMToLLVMConversionPatterns(typeConverter, patterns);
    populateUPMEMToLLVMFinalTypeConversions(typeConverter);
    populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
  }
};

void registerConvertUpmemToLLvmInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *, upmem::UPMEMDialect *dialect) {
    dialect->addInterfaces<UpmemToLlvmDialectInterface>();
  });
}

} // namespace mlir::upmem