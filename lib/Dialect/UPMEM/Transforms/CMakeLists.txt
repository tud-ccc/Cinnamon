add_mlir_dialect_library(UPMEMTransforms
    DedupKernels.cpp
    UpmemInferAccelerator.cpp
    UpmemSimulator.cpp

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
    CnmIR
    CnmTransforms
    MLIRFuncDialect
)

target_link_libraries(UPMEMTransforms PRIVATE dlib)
include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/third-party/dlib)
