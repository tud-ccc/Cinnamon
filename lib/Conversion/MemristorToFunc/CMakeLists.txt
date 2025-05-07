add_mlir_conversion_library(MLIRMemristorToFunc
    MemristorToFunc.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/memristor/Conversion/MemristorToFunc

    DEPENDS
    MemristorConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MemristorIR
    MLIRDialectUtils
    MLIRFuncDialect
    MLIRTransformUtils
    MLIRReconcileUnrealizedCasts
)
