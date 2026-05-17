add_mlir_dialect_library(CinmAcceleratorInference
    AcceleratorInference.cpp
    BananasSearch.cpp

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
    CinmIR
    CinmUtils
)

target_link_libraries(CinmAcceleratorInference PRIVATE pybind11::embed)
target_compile_definitions(CinmAcceleratorInference PRIVATE
    BANANAS_BRIDGE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
