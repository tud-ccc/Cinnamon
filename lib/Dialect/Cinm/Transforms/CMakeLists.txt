add_mlir_dialect_library(CinmTransforms
    AbsorbStaticSlices.cpp
    AssignPlatformsPass.cpp
    InferTileSizesPass.cpp
    BufferizeCinm.cpp
    CinmTransformsUtils.cpp
    CinmMemoryCleanup.cpp
    CinmQuantization.cpp
    CleanupLinalgPass.cpp
    ComputeTransforms.cpp
    CinmTilingImplementations.cpp
    CompleteComputeGraph.cpp
    DecomposeAccumulations.cpp
    DecomposeGemmToGemv.cpp
    ExpandComputeScope.cpp
    GemvMinWrite.cpp
    MergeHostBlocks.cpp
    OutlineHostBlocks.cpp
    RelowerStandaloneOps.cpp
    SimplifyComputeSignatures.cpp
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
