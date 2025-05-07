

add_library(
    UpmemDialectRuntime
    STATIC
    upmem_rt.c
    memref_rt.cpp
)
target_link_libraries(UpmemDialectRuntime PRIVATE Upmem::dpu)
#target_link_libraries(UpmemDialectRuntime PUBLIC mlir_c_runner_utils mlir_runner_utils)