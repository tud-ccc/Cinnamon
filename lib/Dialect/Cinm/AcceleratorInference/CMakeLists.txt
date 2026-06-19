add_mlir_dialect_library(CinmAcceleratorInference
    AcceleratorInference.cpp
    BananasSearch.cpp
    SpaceBuilder.cpp

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
    mlpack::mlpack 
    indicators::indicators
)

# target_link_libraries(CinmAcceleratorInference PUBLIC mlpack::mlpack indicators::indicators)
