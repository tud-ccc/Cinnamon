################################################################################
# check-cinm-mlir
#
# The cinm-mlir regression test project.
################################################################################

project(check-cinm-mlir)

# Configure the testing site configuration.
configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)
set(TEST_DEPENDS
    FileCheck count not
    cinm-opt
)

# Create the test suite.
add_lit_testsuite(${PROJECT_NAME}
    "Running the cinm-mlir regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS
        ${TEST_DEPENDS}
)

add_lit_testsuites(cinmMLIR
    ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS
    ${TEST_DEPENDS}
)

enable_testing()

find_package(GTest REQUIRED)
include(GoogleTest)

add_custom_target(check-unit-tests
    COMMAND ${CMAKE_CTEST_COMMAND}
            --output-on-failure
            -C $<CONFIG>
)

function(add_unittest test_dirname)
  add_executable(${test_dirname} ${ARGN})
  target_link_libraries(${test_dirname} PRIVATE GTest::gtest_main)
  add_dependencies(check-unit-tests ${test_dirname})
  gtest_discover_tests(${test_dirname})
endfunction()


add_subdirectory(Unit)
