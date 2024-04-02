add_mlir_dialect_library(CnmTransforms
    SPIRVAttachAttributes.cpp

DEPENDS
    CnmIncGen
    CnmTransformPassesIncGen

LINK_LIBS PUBLIC
    MLIRGPUDialect
    MLIRIR
    MLIRSPIRVTarget
    MLIRTransformUtils
)
