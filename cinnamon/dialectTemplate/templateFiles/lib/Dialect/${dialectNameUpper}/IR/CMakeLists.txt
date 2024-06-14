add_mlir_dialect_library(${dialectNameUpper}IR
        ${dialectNameUpper}Base.cpp
        ${dialectNameUpper}Ops.cpp
        ${dialectNameUpper}Types.cpp

    DEPENDS
        ${dialectNameUpper}IncGen
        ${dialectNameUpper}ConversionPassIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces
)
