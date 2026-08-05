add_mlir_dialect_library(CnmTransforms
    SPIRVAttachAttributes.cpp
    BufferizeCnm.cpp
    HoistWorkgroups.cpp
    FuseLaunches.cpp
    EnsureScatterGatherContiguous.cpp
    ScatterOptimizations.cpp
    IsolateLinalgCaptures.cpp

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
    MLIRTensorDialect
    MLIRSPIRVTarget
    MLIRTransformUtils
    CnmIR
)
