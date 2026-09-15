################################################################################
# cinm-translate
#
# The cinnamon translation driver.
################################################################################

project(cinm-translate)

add_executable(${PROJECT_NAME}
    cinm-translate.cpp
)

# Link all standard MLIR dialect and translation libs.
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        MLIRIR
        MLIRParser
        MLIRPass
        MLIRTranslateLib
        MLIRSupport
        ${dialect_libs}
        ${translation_libs}
)

# compile_dpu.sh is the reader half of this translator's output contract: it is
# the only thing that parses the `// UPMEM-TRANSLATE:` header emitted by
# UPMEMTranslateToCpp. The two are one protocol and change together, so the
# script ships from here rather than from whichever tree happens to invoke it,
# and lands on the same PATH as cinm-opt and cinm-translate.
set(cinm_compile_dpu ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/cinm-compile-dpu)
add_custom_command(
    OUTPUT ${cinm_compile_dpu}
    COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/compile_dpu.sh ${cinm_compile_dpu}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/compile_dpu.sh
    COMMENT "Staging cinm-compile-dpu"
)
add_custom_target(cinm-compile-dpu ALL DEPENDS ${cinm_compile_dpu})

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
# The script rather than the staged copy: PROGRAMS sets the execute bit, and
# the source is what the custom command above copies anyway.
install(PROGRAMS compile_dpu.sh DESTINATION bin RENAME cinm-compile-dpu)
