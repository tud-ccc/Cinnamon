add_mlir_dialect_library(UPMEMTransforms
    DedupKernels.cpp
    UpmemLegalizeForTranslation.cpp
    UpmemAnnotateCosts.cpp
    UpmemInferAccelerator.cpp
    UpmemOpCountSimulator.cpp
    UpmemPythonSimulator.cpp
    SimulationTemplates.cpp

DEPENDS
    UPMEMIncGen
    UPMEMTransformPassesIncGen

LINK_LIBS PUBLIC
    MLIRIR
    MLIRTransformUtils
    MLIRPass
    UPMEMIR
    CinmIR
    CinmTransforms
    CinmAcceleratorInference
    CnmIR
    CnmTransforms
    MLIRFuncDialect
    MLIRAffineDialect
    MLIRCinmToCnm
    MLIRCnmToUPMEM
    MLIRBufferizationTransforms
    MLIRAffineTransforms
    MLIRMemRefTransforms
    MLIRAffineToStandard
    MLIRLinalgToStandard
    MLIRArithDialect
    MLIRSCFDialect
)

target_link_libraries(UPMEMTransforms PRIVATE UpmemCostModel)
