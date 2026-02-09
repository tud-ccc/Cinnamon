add_mlir_dialect_library(AlpineTransforms
    HoistWriteWeights.cpp
    LowerMVMToStreaming.cpp

DEPENDS
    AlpineIncGen
    AlpineTransformPassesIncGen

LINK_LIBS PUBLIC
    MLIRIR
    CinmIR
    AlpineIR
    MLIRMemRefDialect
    MLIRLLVMDialect
    MLIRAffineDialect
    MLIRLinalgDialect
    MLIRSPIRVTarget
    MLIRTransformUtils
)
