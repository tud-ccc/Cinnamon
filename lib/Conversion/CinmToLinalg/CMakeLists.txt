add_mlir_conversion_library(MLIRCinmToLinalg
    CinmToLinalg.cpp

    DEPENDS
    CinmConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    CinmIR
    MLIRArithDialect
    MLIRLinalgDialect
    MLIRMathDialect
    MLIRTensorDialect
    MLIRTransformUtils
    MLIRReconcileUnrealizedCasts
)
