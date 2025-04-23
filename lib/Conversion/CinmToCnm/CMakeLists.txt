add_mlir_conversion_library(MLIRCinmToCnm
    CinmToCnm.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/cnm/Conversion/CinmToCnm

    DEPENDS
    CinmConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    CnmIR
    CinmUtils
    MLIRDialectUtils
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRTensorDialect
    MLIRTransformUtils
    MLIRReconcileUnrealizedCasts
)
