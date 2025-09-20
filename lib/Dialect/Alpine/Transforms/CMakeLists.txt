add_mlir_dialect_library(AlpineTransforms
    LowerMVMToStreaming.cpp

DEPENDS
    AlpineIncGen
    AlpineTransformPassesIncGen

LINK_LIBS PUBLIC
    MLIRIR
    AlpineIR
    MLIRMemRefDialect
    MLIRLLVMDialect
    MLIRAffineDialect
    MLIRLinalgDialect
    MLIRSPIRVTarget
    MLIRTransformUtils
)
