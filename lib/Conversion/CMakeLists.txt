add_mlir_library(CinmCommonPatterns
    CommonPatterns.cpp

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRDialectUtils
)


add_subdirectory(CnmToGPU)
add_subdirectory(CnmToUPMEM)
add_subdirectory(CinmToCnm)
