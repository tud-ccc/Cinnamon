add_mlir_conversion_library(MLIRCimToAlpine
    CimToAlpine.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/cim/Conversion/CimToAlpine

    DEPENDS
    CimConversionPassIncGen
    AlpineIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    CimIR
    AlpineIR
    MLIRDialectUtils
    MLIRBufferizationDialect
    MLIRMemRefDialect
    MLIRTensorDialect
    MLIRTransformUtils
    MLIRReconcileUnrealizedCasts
)
