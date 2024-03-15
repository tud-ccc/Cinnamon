add_mlir_conversion_library(MLIRCnmToGPU
    CnmToGPU.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/cnm/Conversion/CnmToGPU

    DEPENDS
    CnmConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    CnmIR
    MLIRDialectUtils
    MLIRFuncDialect
    MLIRTransformUtils
    MLIRReconcileUnrealizedCasts
)
