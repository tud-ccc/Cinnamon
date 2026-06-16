find_package(PkgConfig REQUIRED)
pkg_check_modules(mlpack REQUIRED IMPORTED_TARGET mlpack)

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

target_link_libraries(CinmAcceleratorInference PRIVATE PkgConfig::mlpack indicators::indicators)
