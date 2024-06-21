add_mlir_dialect_library(CnmIR
        CnmBase.cpp
        CnmOps.cpp
        CnmTypes.cpp

    DEPENDS
        CnmIncGen
        CnmConversionPassIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces
        CinmUtils
)
