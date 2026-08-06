
add_mlir_dialect_library(CinmUtils
    CinmUtils.cpp
    Permutation.cpp
    SchedulingSupport.cpp

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRAffineDialect
        MLIRSCFTransforms
        MLIRDialectUtils
        MLIRLinalgDialect
        MLIRTensorDialect
        MLIRMemRefDialect
        MLIRTransformUtils
        MLIRPass
)
