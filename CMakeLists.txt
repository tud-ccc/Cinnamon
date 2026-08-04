################################################################################
# cinm-mlir
#
################################################################################

cmake_minimum_required(VERSION 3.28)

# Add custom modules to the search path.
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

project(cinm-mlir
    VERSION     0.2.0
    LANGUAGES   C CXX
)

option(CINM_BUILD_GPU_SUPPORT "Enable vulkan runner and GPU libs" OFF)
option(TORCH_MLIR_DIR "Directory of the Torch-MLIR installation" "")
option(UPMEM_DIR "Directory of the UPMEM installation" "")

if (CINM_BUILD_GPU_SUPPORT)
    add_compile_definitions(CINM_GPU_SUPPORT)
endif()

if (TORCH_MLIR_DIR)
    add_compile_definitions(CINM_TORCH_MLIR_ENABLED)

    set(TORCH_MLIR_INCLUDE_DIR "${TORCH_MLIR_DIR}/include")
    set(TORCH_MLIR_LIB_DIR "${TORCH_MLIR_DIR}/lib")
    include_directories(SYSTEM ${TORCH_MLIR_INCLUDE_DIR})
    link_directories(${TORCH_MLIR_LIB_DIR})

    file(GLOB TORCH_MLIR_LIBS "${TORCH_MLIR_LIB_DIR}/*.a")

    get_property(CINM_DEPENDENCY_LIBS GLOBAL PROPERTY CINM_DEPENDENCY_LIBS)
    list(APPEND CINM_DEPENDENCY_LIBS ${TORCH_MLIR_LIBS})
    set_property(GLOBAL PROPERTY CINM_DEPENDENCY_LIBS ${CINM_DEPENDENCY_LIBS})

else()
    message(WARNING "TORCH_MLIR_DIR not set, torch frontend wont be available.")
endif()

# Set global language standard to C++20.
set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard to conform to")

# Detect if this is a stand-alone build.
if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
    # Set shared output directories.
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
endif()

# Fix for MLIR:
# CMP0116: Ninja generators transform `DEPFILE`s from `add_custom_command()`
# New in CMake 3.20. https://cmake.org/cmake/help/latest/policy/CMP0116.html
if(POLICY CMP0116)
    cmake_policy(SET CMP0116 OLD)
endif()

# Find MLIR.
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Load LLVM and MLIR CMake modules.
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Apply target properties for LLVM/MLIR globally.
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
# Note: do NOT call add_definitions(${LLVM_DEFINITIONS}) here. HandleLLVMOptions already
# adds all required compile definitions via add_compile_definitions(). LLVM_DEFINITIONS is
# a space-separated string (not a CMake list), so passing it to add_definitions as a single
# argument creates a malformed -D_GLIBCXX_USE_CXX11_ABI="0 -D..." flag.

# Apply target properties for this project globally.
include_directories(SYSTEM ${PROJECT_BINARY_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include)

add_compile_options(
    # NOTE: C++20 warning adjustments for this project.
    -Wall
    -Wextra
    -Wno-ambiguous-reversed-operator
    # NOTE: GDB < 10.x can't handle DWARFv5 correctly.
    $<$<CXX_COMPILER_ID:Clang>:-fdebug-default-version=4>
    $<$<CXX_COMPILER_ID:Clang>:-fcolor-diagnostics>
)

# The vectorized constraint evaluation in Dialect/Cinm/AcceleratorInference
# leans on the compiler auto-vectorizing integer elementwise work. The
# x86-64 baseline has no SIMD 64-bit integer compare at all (pcmpgtq arrived
# with SSE4.2), so without this the constraint scan compiles to scalar code.
#
# x86-64-v3 (AVX2; Intel Haswell 2013+, AMD Excavator+) is the default rather
# than "native" deliberately: it is a documented microarchitecture level, so
# the resulting binary is reproducible and still runs on any other v3 host,
# whereas "native" bakes in whatever this particular build machine happens to
# support. Set to "native" to squeeze out this host only, or to "" to disable
# and go back to the portable baseline.
set(CINM_X86_ARCH "x86-64-v3" CACHE STRING
    "Value passed to -march= on x86-64 hosts; empty disables the flag")

if (CINM_X86_ARCH AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    include(CheckCXXCompilerFlag)
    # Key the cache entry on the value, so changing CINM_X86_ARCH re-runs the
    # check instead of reusing the previous value's verdict.
    string(MAKE_C_IDENTIFIER "CINM_HAVE_MARCH_${CINM_X86_ARCH}" _march_ok_var)
    check_cxx_compiler_flag("-march=${CINM_X86_ARCH}" ${_march_ok_var})
    if (${_march_ok_var})
        add_compile_options(-march=${CINM_X86_ARCH})
        message(STATUS "Building with -march=${CINM_X86_ARCH}")
    else()
        message(WARNING "Compiler rejects -march=${CINM_X86_ARCH}; "
                        "building without it (constraint scan stays scalar).")
    endif()
    unset(_march_ok_var)
endif()

get_property(LTY GLOBAL PROPERTY LINKER_TYPE)
if (LTY STREQUAL "LLD")
    add_link_options(--for-linker --color-diagnostics=always)
endif()

if (TORCH_MLIR_DIR)
    set_property(GLOBAL PROPERTY TORCH_MLIR_ENABLED ON)
    set(TORCH_MLIR_INCLUDE_DIR "${TORCH_MLIR_DIR}/include")
    # include_directories(SYSTEM ${TORCH_MLIR_INCLUDE_DIR})

    add_library(Torch::TorchMLIRTorchDialect STATIC IMPORTED GLOBAL)
    set_property(TARGET Torch::TorchMLIRTorchDialect PROPERTY
                 IMPORTED_LOCATION ${TORCH_MLIR_DIR}/lib/libTorchMLIRTorchDialect.a)
    add_library(Torch::TorchMLIRTorchUtils STATIC IMPORTED GLOBAL)
    set_property(TARGET Torch::TorchMLIRTorchUtils PROPERTY
                 IMPORTED_LOCATION ${TORCH_MLIR_DIR}/lib/libTorchMLIRTorchUtils.a)
    add_library(Torch::TorchMLIRTorchConversionDialect STATIC IMPORTED GLOBAL)
    set_property(TARGET Torch::TorchMLIRTorchConversionDialect PROPERTY
                 IMPORTED_LOCATION ${TORCH_MLIR_DIR}/lib/libTorchMLIRTorchConversionDialect.a)

    add_library(Torch::All INTERFACE IMPORTED)
    set_property(TARGET Torch::All PROPERTY
      INTERFACE_LINK_LIBRARIES Torch::TorchMLIRTorchDialect
            Torch::TorchMLIRTorchUtils
            Torch::TorchMLIRTorchConversionDialect)
    target_include_directories(Torch::All INTERFACE ${TORCH_MLIR_INCLUDE_DIR})

else()
    set_property(GLOBAL PROPERTY TORCH_MLIR_ENABLED OFF)
    message(WARNING "TORCH_MLIR_DIR not set")
endif()

if (UPMEM_DIR)
    add_library(Upmem::dpu SHARED IMPORTED GLOBAL)
    set_property(TARGET Upmem::dpu PROPERTY
                 IMPORTED_LOCATION ${UPMEM_DIR}/lib/libdpu.so)
    target_include_directories(Upmem::dpu INTERFACE ${UPMEM_DIR}/include/dpu)
    add_subdirectory(runtime/Upmem)
else()
    message(WARNING "UPMEM_DIR not set, runtime lib will not be compiled.")
endif()

# These are installed by conan:
#     conan install . --output-folder=build --build=missing
find_package(indicators REQUIRED)
find_package(mlpack REQUIRED)

# LLVM's llvm_add_library wraps each target in an OBJECT library and
# re-forwards includes via target_include_directories without SYSTEM
# (AddLLVM.cmake ~655), stripping the -isystem marker that CMake would
# normally apply to IMPORTED targets. Workaround: add conan include paths
# as global system includes so the flag reaches every compilation unit.
string(TOUPPER "${CMAKE_BUILD_TYPE}" _build_type_upper)
foreach(_pkg mlpack indicators ensmallen cereal armadillo OpenBLAS stb)
  set(_var "${_pkg}_INCLUDE_DIRS_${_build_type_upper}")
  if(DEFINED "${_var}")
    include_directories(SYSTEM "${${_var}}")
  elseif(DEFINED "${_pkg}_INCLUDE_DIRS")
    message(SEND_ERROR "Package '${_pkg}' has no include dirs for build type '${CMAKE_BUILD_TYPE}' "
      "(looked for '${_var}'). Conan dependencies were likely built with a different build type. "
      "Re-run: conan install . --output-folder=build --build=missing -s build_type=${CMAKE_BUILD_TYPE}")
  endif()
endforeach()
unset(_build_type_upper)
unset(_var)
# unset(_pkg)


add_subdirectory("third-party/upmem-cost-model")


add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(tools)

add_subdirectory(runtime/Memristor)
