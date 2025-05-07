add_mlir_dialect_library(CinmTransforms
    SoftmaxToCinmPass.cpp
    TilingPass.cpp

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
