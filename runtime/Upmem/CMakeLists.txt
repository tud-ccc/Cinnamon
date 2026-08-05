

add_library(
    UpmemDialectRuntime
    STATIC
    upmem_rt.c
    timers.c
    memref_rt.cpp
)
target_link_libraries(UpmemDialectRuntime PRIVATE Upmem::dpu)
target_compile_definitions(UpmemDialectRuntime PUBLIC UPMEM_RT_STATS)
#target_link_libraries(UpmemDialectRuntime PUBLIC mlir_c_runner_utils mlir_runner_utils)
