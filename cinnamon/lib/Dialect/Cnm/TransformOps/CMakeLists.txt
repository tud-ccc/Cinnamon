add_mlir_dialect_library(CnmTransformDialectExtension
    CnmTransformOps.cpp
    CnmTransformExtension.cpp

DEPENDS
    CnmIncGen
    CnmTransformPassesIncGen

LINK_LIBS PUBLIC
    MLIRIR
    MLIRLLVMDialect
    MLIRAffineDialect
    MLIRLinalgDialect
    MLIRTransformUtils
    MLIRTransformDialect
    MLIRTransformDialectUtils
)
