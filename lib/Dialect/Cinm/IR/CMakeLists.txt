add_mlir_dialect_library(CinmIR
        CinmBase.cpp
        CinmOps.cpp
        CinmTypes.cpp

    DEPENDS
        CinmIncGen
        CinmConversionPassIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces
        CinmInterfaces
)
