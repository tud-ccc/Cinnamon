

add_library(
    UpmemDialectRuntime
    STATIC
    upmem_rt.c
    timers.c
    memref_rt.cpp
)
target_link_libraries(UpmemDialectRuntime PRIVATE Upmem::dpu)
target_compile_definitions(UpmemDialectRuntime PUBLIC UPMEM_RT_STATS)

# The bench drivers compile against these headers and link this archive, so
# both have to reach the install prefix: the drivers are built outside this
# project, by the experiment pipelines, against whatever prefix is active.
install(TARGETS UpmemDialectRuntime ARCHIVE DESTINATION lib)
install(FILES upmem_rt.h timers.h DESTINATION include/cinm-mlir/runtime/Upmem)
#target_link_libraries(UpmemDialectRuntime PUBLIC mlir_c_runner_utils mlir_runner_utils)
