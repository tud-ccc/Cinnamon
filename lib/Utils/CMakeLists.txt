
add_mlir_dialect_library(CinmUtils
    CinmUtils.cpp
    DebugPasses.cpp
    SchedulingSupport.cpp

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRDialectUtils
        MLIRTensorDialect
        MLIRMemRefDialect
        MLIRTransformUtils
        MLIRPass
)