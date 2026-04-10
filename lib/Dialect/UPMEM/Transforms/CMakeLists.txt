add_mlir_dialect_library(UPMEMTransforms
    DedupKernels.cpp

DEPENDS
    UPMEMIncGen
    UPMEMTransformPassesIncGen

LINK_LIBS PUBLIC
    MLIRIR
    MLIRTransformUtils
    UPMEMIR
    MLIRFuncDialect
)
