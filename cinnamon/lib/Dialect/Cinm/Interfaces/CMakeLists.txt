add_mlir_library(CinmInterfaces
        TilingInterface.cpp

    ENABLE_AGGREGATION
    DEPENDS
        CinmIncGen

    LINK_LIBS PUBLIC
    CinmCommonPatterns
)
