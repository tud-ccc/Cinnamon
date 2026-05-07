add_mlir_dialect_library(CinmTransforms
    AnnotateTileSizesPass.cpp
    InferTileSizesPass.cpp
    BufferizeCinm.cpp
    CinmMemoryCleanup.cpp
    CinmQuantization.cpp
    CleanupLinalgPass.cpp
    ComputeTransforms.cpp
    CostModel.cpp
    DecomposeAccumulations.cpp
    DecomposeGemmToGemv.cpp
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
    MLIRLLVMCommonConversion
    MLIRLLVMDialect
    MLIRTransformDialectUtils
    CinmIR
)

target_link_libraries(CinmTransforms PRIVATE pybind11::embed dlib::dlib)
