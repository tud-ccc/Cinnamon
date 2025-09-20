add_mlir_conversion_library(MLIRLinalgToCinm
    Im2ColToMatmul.cpp
    LinalgToCinm.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/cim/Conversion/LinalgToCinm

    DEPENDS
    CinmConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    CinmUtils
    CinmIR
    MLIRDialectUtils
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRTensorDialect
    MLIRTransformUtils
    MLIRReconcileUnrealizedCasts
)
