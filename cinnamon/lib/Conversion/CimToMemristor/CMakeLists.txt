add_mlir_conversion_library(MLIRCimToMemristor
    CimToMemristor.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/cim/Conversion/CimToMemristor

    DEPENDS
    CimConversionPassIncGen
    MemristorIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    CimIR
    MLIRDialectUtils
    MLIRBufferizationDialect
    MLIRMemRefDialect
    MLIRTensorDialect
    MLIRTransformUtils
    MLIRReconcileUnrealizedCasts
)
