add_mlir_conversion_library(MLIRCinmToCim
    CinmToCim.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/cim/Conversion/CinmToCim

    DEPENDS
    CinmConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    CimIR
    CinmUtils
    MLIRDialectUtils
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRTensorDialect
    MLIRTransformUtils
    MLIRReconcileUnrealizedCasts
)
