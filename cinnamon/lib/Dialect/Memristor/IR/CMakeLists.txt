add_mlir_dialect_library(MemristorIR
        MemristorBase.cpp
        MemristorOps.cpp
        MemristorTypes.cpp

    DEPENDS
        MemristorIncGen
        # MemristorConversionPassIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces
)
