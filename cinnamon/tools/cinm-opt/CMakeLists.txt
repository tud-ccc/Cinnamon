################################################################################
# cinm-opt
#
# The cinm-mlir optimizer driver.
################################################################################

project(cinm-opt)

add_executable(${PROJECT_NAME}
    cinm-opt.cpp
)

# Link all standard MLIR dialect and conversion libs.
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(dependency_libs GLOBAL PROPERTY CINM_DEPENDENCY_LIBS)
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        -Wl,--start-group
        MLIROptLib
        ${dialect_libs}
        ${conversion_libs}
        ${extension_libs}
        ${dependency_libs}
        LLVMSupport
        -Wl,--end-group
)
