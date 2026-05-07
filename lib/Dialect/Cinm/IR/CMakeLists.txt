add_mlir_dialect_library(CinmIR
        CinmBase.cpp
        CinmOps.cpp
        CinmTypes.cpp
        CinmTilingFactors.cpp
        CinmTilingImplementations.cpp
        TilingInterface.cpp
        CinmUtils.cpp

    DEPENDS
        CinmIncGen
        CinmConversionPassIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces
        MLIRArithDialect
        MLIRAffineDialect
        MLIRTensorDialect
        MLIRLinalgUtils
)
