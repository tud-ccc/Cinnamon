project(cinm-vulkan-runner)

find_package(Vulkan)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    MLIRAnalysis
    MLIRBuiltinToLLVMIRTranslation
    MLIRExecutionEngine
    MLIRIR
    MLIRJitRunner
    MLIRLLVMToLLVMIRTranslation
    MLIRParser
    MLIRSupport
    MLIRTargetLLVMIRExport
    MLIRTransforms
    MLIRTranslateLib
    ${Vulkan_LIBRARY}
)

add_executable(cinm-vulkan-runner cinm-vulkan-runner.cpp)
llvm_update_compile_flags(cinm-vulkan-runner)
target_link_libraries(cinm-vulkan-runner PRIVATE ${LIBS} ${dialect_libs} ${conversion_libs})
