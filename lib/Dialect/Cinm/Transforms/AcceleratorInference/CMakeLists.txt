add_mlir_dialect_library(CinmAcceleratorInference
    AcceleratorInference.cpp

DEPENDS
    CinmIncGen

LINK_LIBS PUBLIC
    MLIRIR
    CinmIR
    CinmUtils
)

target_link_libraries(CinmAcceleratorInference PRIVATE dlib)
include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/third-party/dlib)
