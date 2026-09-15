add_mlir_dialect_library(CnmIR
        CnmBase.cpp
        CnmOps.cpp
        CnmScatterMap.cpp
        CnmTypes.cpp
        CnmInterfaces.cpp

    DEPENDS
        CnmIncGen
        CinmIncGen
        CnmConversionPassIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces
        MLIRInferTypeOpInterface
        MLIRDestinationStyleOpInterface
        CinmUtils
        CinmIR
)
