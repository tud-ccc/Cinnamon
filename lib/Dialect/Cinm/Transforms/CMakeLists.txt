add_subdirectory(AcceleratorInference)

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
    MLIRLinalgUtils
    MLIRTensorDialect
    MLIRLLVMCommonConversion
    MLIRLLVMDialect
    MLIRTransformDialectUtils
    CinmIR
)

target_link_libraries(CinmTransforms PRIVATE pybind11::embed dlib)
include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/third-party/dlib)
