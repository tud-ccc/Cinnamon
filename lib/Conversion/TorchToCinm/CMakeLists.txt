add_mlir_conversion_library(MLIRTorchToCinm
    TorchToCinm.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/cinm_frontend/Conversion/TorchToCinm

    DEPENDS
    CinmFrontendConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    CinmIR
    CinmUtils
    MLIRDialectUtils
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRTensorDialect
    MLIRTransformUtils
    MLIRReconcileUnrealizedCasts
    Torch::All
)
