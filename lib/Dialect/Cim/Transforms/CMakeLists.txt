add_mlir_dialect_library(CimTransforms
    SchedulingPasses.cpp
    MarkRelowerPass.cpp
    CimCleanupUnsupported.cpp

DEPENDS
    CimIncGen
    CimTransformPassesIncGen

LINK_LIBS PUBLIC
    MLIRIR
    MLIRMemRefDialect
    MLIRLLVMDialect
    MLIRAffineDialect
    MLIRLinalgDialect
    MLIRSPIRVTarget
    MLIRTransformUtils
    CimIR
)
