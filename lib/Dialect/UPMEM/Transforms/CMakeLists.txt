add_mlir_dialect_library(UPMEMTransforms
    DedupKernels.cpp
    HoistLoadPrograms.cpp
    UpmemAsyncLaunches.cpp
    UpmemLegalizeForTranslation.cpp
    UpmemAlignLocalTransfers.cpp
    UpmemAnnotateCosts.cpp
    UpmemCheckOccupancy.cpp
    UpmemCoalesceLocalTransfers.cpp
    UpmemInferAccelerator.cpp
    UpmemOpCountSimulator.cpp
    UpmemPythonSimulator.cpp
    UpmemRegisterTileLoops.cpp
    UpmemTileMRAMBuffers.cpp
    SpecializeTransfers.cpp

DEPENDS
    UPMEMIncGen
    UPMEMTransformPassesIncGen

LINK_LIBS PUBLIC
    MLIRIR
    MLIRAnalysis
    MLIRMemRefDialect
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
    MLIRAffineAnalysis
    MLIRAffineUtils
    MLIRMemRefTransforms
    MLIRAffineToStandard
    MLIRLinalgToStandard
    MLIRArithDialect
    MLIRSCFDialect
    MLIRLinalgTransforms
    MLIRSCFTransforms
)

target_link_libraries(UPMEMTransforms PRIVATE UpmemCostModel)
