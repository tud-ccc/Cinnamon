add_mlir_conversion_library(MLIRUPMEMToLLVM
    UPMEMToLLVM.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/upmem/Conversion/UPMEMToLLVM

    DEPENDS
    UPMEMConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    CinmCommonPatterns
    UPMEMIR
    MLIRDialectUtils
    MLIRFuncDialect
    MLIRTransformUtils
    MLIRReconcileUnrealizedCasts
)
