add_mlir_dialect_library(UPMEMIR
        UPMEMBase.cpp
        UPMEMOps.cpp
        UPMEMTypes.cpp 
        UPMEMAttributes.cpp
        UpmemxBtflScheduling.cpp
        UpmemxBtflLowerings.cpp
        InferIntRangeInterfaceImpls.cpp
        
    DEPENDS
        UPMEMIncGen
        UPMEMConversionPassIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces

        MLIRArithDialect
        MLIRDLTIDialect
        MLIRFunctionInterfaces
        MLIRInferIntRangeInterface
        MLIRMemRefDialect
        MLIRSupport

        TileFirst::TileFirstIR
        TileFirst::BtflIR
        TileFirst::BtflTransforms
        TileFirst::ThreadsIR
)
