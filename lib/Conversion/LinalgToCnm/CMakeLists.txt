add_mlir_conversion_library(MLIRLinalgToCnm
    LinalgToCnm.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/cnm/Conversion/LinalgToCnm

    DEPENDS
    CnmConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    CinmCommonPatterns
    CnmIR
    CinmIR
    CinmUtils
    MLIRDialectUtils
    MLIRLinalgDialect
    MLIRTensorDialect
    MLIRTransformUtils
)
