add_mlir_dialect_library(CnmTransforms
    SPIRVAttachAttributes.cpp
    BufferizeCnm.cpp
    HoistWorkgroups.cpp
    FuseLaunches.cpp
    EnsureScatterGatherContiguous.cpp
    ScatterOptimizations.cpp
    IsolateLinalgCaptures.cpp
    PromoteHostAllocs.cpp

DEPENDS
    CnmIncGen
    CinmIncGen
    CnmTransformPassesIncGen

LINK_LIBS PUBLIC
    MLIRIR
    MLIRGPUDialect
    MLIRBufferizationDialect
    MLIRMemRefDialect
    MLIRFuncDialect
    MLIRLLVMDialect
    MLIRAffineDialect
    MLIRLinalgDialect
    MLIRTensorDialect
    MLIRSPIRVTarget
    MLIRTransformUtils
    CnmIR
    CinmIR
)
