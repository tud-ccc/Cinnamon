################################################################################
# cinm-unit-tests
#
# Unit tests for the parts of cinm-mlir that lit cannot reach: code with no IR
# on either side of it, where a test has to call the API and look at what comes
# back. The search space is the case that forced this -- what it *contains* is
# not observable from a pass pipeline, only what the search eventually picks.
#
# Plain executables rather than gtest: the vendored LLVM here is built without
# LLVM_INCLUDE_TESTS, so there is no llvm_gtest target to link, and adding a
# dependency to make an assertion macro available is a poor trade. Each test
# reports its own failures and exits non-zero; lit runs it (see test/Unit) so
# `check-cinm-mlir` stays the single command.
################################################################################


add_unittest(CinmUnitTests
  ConstraintIRTest.cpp
)
target_link_libraries(CinmUnitTests
    PRIVATE
        CinmAcceleratorInference
        MLIRIR
        MLIRSupport
        LLVMSupport
)
