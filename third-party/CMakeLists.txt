
set(UPMEM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/upmem)
add_library(Upmem::dpu SHARED IMPORTED GLOBAL)
set_property(TARGET Upmem::dpu PROPERTY
             IMPORTED_LOCATION ${UPMEM_DIR}/lib/dpu.so)
target_include_directories(Upmem::dpu INTERFACE ${UPMEM_DIR}/include/dpu)