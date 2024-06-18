add_mlir_dialect_library(UPMEMTransforms
    OutlineFunction.cpp
    DedupKernels.cpp

DEPENDS
    UPMEMIncGen
    UPMEMTransformPassesIncGen

LINK_LIBS PUBLIC
    MLIRIR
    MLIRTransformUtils
)
