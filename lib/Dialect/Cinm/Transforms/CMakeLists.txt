add_mlir_dialect_library(CinmTransforms
    AssignPlatformsPass.cpp
    InferTileSizesPass.cpp
    BufferizeCinm.cpp
    CinmTransformsUtils.cpp
    CinmMemoryCleanup.cpp
    CinmQuantization.cpp
    CleanupLinalgPass.cpp
    ComputeTransforms.cpp
    CinmTilingImplementations.cpp
    DecomposeAccumulations.cpp
    DecomposeGemmToGemv.cpp
    ExpandComputeScope.cpp
    GemvMinWrite.cpp
    RelowerStandaloneOps.cpp
    SoftmaxToCinmPass.cpp
    TilingPass.cpp

DEPENDS
    CinmIncGen
    CinmTransformPassesIncGen

LINK_LIBS PUBLIC
    MLIRIR
    MLIRAffineDialect
    MLIRLinalgDialect
    MLIRLinalgUtils
    MLIRTensorDialect
    MLIRLLVMCommonConversion
    MLIRLLVMDialect
    MLIRTransformDialectUtils
    CinmIR
)
