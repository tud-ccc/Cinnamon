################################################################################
# cinm-unit-tests
#
# Unit tests for the parts of cinm-mlir that lit cannot reach: code with no IR
# on either side of it, where a test has to call the API and look at what comes
# back. The search space is the case that forced this -- what it *contains* is
# not observable from a pass pipeline, only what the search eventually picks.
#
################################################################################


add_unittest(CinmUnitTests
  ConstraintIRTest.cpp
  PermutationTest.cpp
  SplitDerivationTest.cpp
)
target_link_libraries(CinmUnitTests
    PRIVATE
        CinmAcceleratorInference
        MLIRIR
        MLIRSupport
        LLVMSupport
)
