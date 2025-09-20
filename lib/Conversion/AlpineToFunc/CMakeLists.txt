add_mlir_conversion_library(MLIRAlpineToFunc
    AlpineToFunc.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/alpine/Conversion/AlpineToFunc

    DEPENDS
    AlpineConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    AlpineIR
    MLIRDialectUtils
    MLIRFuncDialect
    MLIRTransformUtils
    MLIRReconcileUnrealizedCasts
)
