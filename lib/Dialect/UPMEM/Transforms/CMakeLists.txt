add_mlir_dialect_library(UPMEMTransforms
    DedupKernels.cpp
    UpmemAnnotateCosts.cpp
    UpmemInferAccelerator.cpp
    UpmemOpCountSimulator.cpp
    UpmemPythonSimulator.cpp

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

target_link_libraries(UPMEMTransforms PRIVATE dlib pybind11::embed)
include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/third-party/dlib)
add_compile_definitions(
    UPMEM_SIM_PACKAGE_DIR="${CMAKE_SOURCE_DIR}/third-party/upmem-cost-model/packages/upmem_simulator"
)
