add_mlir_dialect_library(CnmTransforms
    SPIRVAttachAttributes.cpp
    BufferizeCnm.cpp
    HoistWorkgroups.cpp
    EnsureScatterGatherContiguous.cpp

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
    CnmIR
)
