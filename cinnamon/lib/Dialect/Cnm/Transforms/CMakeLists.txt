add_mlir_dialect_library(CnmTransforms
    SPIRVAttachAttributes.cpp
    Bufferize.cpp
    HoistWorkgroups.cpp

DEPENDS
    CnmIncGen
    CnmTransformPassesIncGen

LINK_LIBS PUBLIC
    MLIRIR
    MLIRGPUDialect
    MLIRBufferizationDialect
    MLIRMemRefDialect
    MLIRLLVMDialect
    MLIRAffineDialect
    MLIRLinalgDialect
    MLIRSPIRVTarget
    MLIRTransformUtils
)
