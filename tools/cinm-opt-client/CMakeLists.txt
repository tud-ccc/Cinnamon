################################################################################
# cinm-opt-client
#
# Runs cinm-opt invocations in a cinm-opt fork server. Plain C and not linked
# against LLVM, so that starting it is cheap.
################################################################################

project(cinm-opt-client C)

add_executable(${PROJECT_NAME}
    cinm-opt-client.c
)
set_target_properties(${PROJECT_NAME} PROPERTIES C_STANDARD 11)
target_include_directories(${PROJECT_NAME}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../cinm-opt
)
# Needs cinm-opt next to it at runtime.
add_dependencies(${PROJECT_NAME} cinm-opt)

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
