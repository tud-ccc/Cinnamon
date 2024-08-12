add_mlir_dialect_library(HbmpimIR
        HbmpimBase.cpp
        HbmpimOps.cpp
        HbmpimTypes.cpp

    DEPENDS
        HbmpimIncGen
        # HbmpimConversionPassIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces
        CinmUtils
)
