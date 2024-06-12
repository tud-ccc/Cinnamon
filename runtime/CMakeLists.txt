

add_library(
    UpmemDialectRuntime
    STATIC
    upmem_rt.c
)
target_link_libraries(UpmemDialectRuntime PRIVATE Upmem::dpu)