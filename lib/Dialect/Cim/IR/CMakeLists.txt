add_mlir_dialect_library(CimIR
        CimBase.cpp
        CimOps.cpp
        CimTypes.cpp

    DEPENDS
        CimIncGen
        # CimConversionPassIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces
        MLIRInferTypeOpInterface
        CinmUtils
)
