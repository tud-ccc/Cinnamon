add_mlir_dialect_library(CimTransforms
    dummy.cpp

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
)
