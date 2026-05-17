add_mlir_dialect_library(CinmAcceleratorInference
    AcceleratorInference.cpp
    CostModel.cpp

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

target_link_libraries(CinmAcceleratorInference PRIVATE pybind11::embed dlib)
include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/third-party/dlib)
