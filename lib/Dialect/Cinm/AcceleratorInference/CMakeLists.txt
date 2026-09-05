add_mlir_dialect_library(CinmAcceleratorInference
    AcceleratorInference.cpp
    BananasSearch.cpp
    Progress.cpp
    ConfigSpace.cpp
    ConstraintIR.cpp
    ConstraintGecode.cpp
    FusionEdges.cpp
    GraphAllocation.cpp
    GraphInference.cpp
    SpaceBuilder.cpp

DEPENDS
    CinmIncGen
    CinmTransformPassesIncGen

LINK_LIBS PUBLIC
    MLIRIR
    MLIRAffineDialect
    MLIRLinalgDialect
    MLIRLinalgUtils
    MLIRTensorDialect
    MLIRLLVMCommonConversion
    MLIRLLVMDialect
    CinmIR
    CinmUtils
    mlpack::mlpack
    indicators::indicators
    Gecode::gecodeminimodel
    Gecode::gecodesearch
    Gecode::gecodeint
    Gecode::gecodekernel
    Gecode::gecodesupport
)

# target_link_libraries(CinmAcceleratorInference PUBLIC mlpack::mlpack indicators::indicators)
