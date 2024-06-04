add_mlir_conversion_library(MLIRCnmToUPMEM
    CnmToUPMEM.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/cnm/Conversion/CnmToUPMEM

    DEPENDS
    CnmConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    CinmCommonPatterns
    CnmIR
    MLIRDialectUtils
    MLIRFuncDialect
    MLIRTransformUtils
    MLIRReconcileUnrealizedCasts
)
