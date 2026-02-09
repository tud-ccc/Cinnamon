add_mlir_dialect_library(AlpineIR
        AlpineBase.cpp
        AlpineOps.cpp
        AlpineTypes.cpp

    DEPENDS
        AlpineIncGen
        # AlpineConversionPassIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRTensorDialect
        MLIRParser
        MLIRSideEffectInterfaces
)
