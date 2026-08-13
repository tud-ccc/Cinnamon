add_mlir_dialect_library(UPMEMTransforms
    DedupKernels.cpp
    HoistLoadPrograms.cpp
    UpmemLegalizeForTranslation.cpp
    UpmemAnnotateCosts.cpp
    UpmemCheckOccupancy.cpp
    UpmemInferAccelerator.cpp
    UpmemOpCountSimulator.cpp
    UpmemPythonSimulator.cpp
    UpmemTileMRAMBuffers.cpp
    SpecializeTransfers.cpp

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
    MLIRTensorToLinalg
    MLIRCinmToLinalg
    MLIRLinalgToCnm
    MLIRCnmToUPMEM
    MLIRBufferizationTransforms
    MLIRBufferizationPipelines
    MLIRAffineTransforms
    MLIRMemRefTransforms
    MLIRAffineToStandard
    MLIRLinalgToStandard
    MLIRArithDialect
    MLIRSCFDialect
    MLIRLinalgTransforms
    MLIRSCFTransforms
)

target_link_libraries(UPMEMTransforms PRIVATE UpmemCostModel)
