add_mlir_dialect_library(CinmTransforms
    SoftmaxToCinmPass.cpp
    TilingPass.cpp
    AnnotateTileSizesPass.cpp
    LoopInterchangePass.cpp
    CinmBufferizableOpInterface.cpp
    DecomposeAccumulations.cpp
    DecomposeGemmToGemv.cpp
    CinmQuantization.cpp
    CleanupLinalgPass.cpp
    RelowerStandaloneOps.cpp
    CinmMemoryCleanup.cpp

DEPENDS
    CinmIncGen
    CinmTransformPassesIncGen

LINK_LIBS PUBLIC
    MLIRIR
    MLIRLLVMDialect
    MLIRAffineDialect
    MLIRLinalgDialect
    MLIRTransformUtils
    CinmIR
    MLIRLLVMCommonConversion
)
