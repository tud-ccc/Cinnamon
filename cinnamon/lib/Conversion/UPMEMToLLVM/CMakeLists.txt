add_mlir_conversion_library(MLIRUPMEMToLLVM
    UPMEMToLLVM.cpp
    ToLLVMInterfaceImpl.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/upmem/Conversion/UPMEMToLLVM

    DEPENDS
    UPMEMConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    CinmCommonPatterns
    UPMEMIR
    CinmUtils
    MLIRDialectUtils
    MLIRReconcileUnrealizedCasts
    MLIRLLVMDialect
    MLIRLLVMCommonConversion
)
