add_mlir_dialect_library(HbmpimTransforms
    OutlineFunctions.cpp

DEPENDS
    HbmpimIncGen
    HbmpimTransformPassesIncGen

LINK_LIBS PUBLIC
    MLIRIR
    MLIRTransformUtils
)
