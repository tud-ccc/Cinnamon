add_mlir_dialect_library(CimTransforms
    SchedulingPasses.cpp

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
