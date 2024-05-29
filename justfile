# Build recipes for this project.
#

# Load environment vars from .env file
# Write LLVM_BUILD_DIR="path" into that file or set this env var in your shell.
set dotenv-load := true

# Make sure your LLVM is tags/llvmorg-18.1.6

llvm_prefix := env_var("LLVM_BUILD_DIR")
build_type := env_var_or_default("LLVM_BUILD_TYPE", "RelWithDebInfo")
linker := env_var_or_default("CMAKE_LINKER_TYPE", "DEFAULT")
build_dir := "build"

# execute cmake -- this is only needed on the first build
cmake:
    cmake -S . -B {{build_dir}} \
        -G Ninja \
        -DCMAKE_BUILD_TYPE={{build_type}} \
        -DLLVM_DIR={{llvm_prefix}}/lib/cmake/llvm \
        -DMLIR_DIR={{llvm_prefix}}/lib/cmake/mlir \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_LINKER_TYPE={{linker}} \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_C_USING_LINKER_mold=-fuse-ld=mold \
        -DCMAKE_CXX_USING_LINKER_mold=-fuse-ld=mold \

# execute a specific ninja target
doNinja *ARGS:
    ninja -C{{build_dir}} {{ARGS}}


# run build --first build needs cmake though
build: doNinja

alias b := build

# run tests
test: (doNinja "check-cinm-mlir")

[no-cd]
cinm-opt *ARGS:
    {{source_directory()}}/{{build_dir}}/bin/cinm-opt {{ARGS}}

cinm-opt-help: (cinm-opt "--help")

# Start a gdb session on cinm-opt.
debug-cinm-opt *ARGS:
    gdb --args {{build_dir}}/bin/cinm-opt {{ARGS}}

cinm-vulkan-runner FILE *ARGS:
    {{build_dir}}/bin/cinm-vulkan-runner {{FILE}} \
        --shared-libs=../llvm-project/build/lib/libvulkan-runtime-wrappers.so,../llvm-project/build/lib/libmlir_runner_utils.so.17 \
        {{ARGS}}

# Invoke he LLVM IR compiler.
llc *ARGS:
    {{llvm_prefix}}/bin/llc {{ARGS}}

# Lowers Sigi all the way to LLVM IR. Temporary files are left there.
llvmDialectIntoExecutable FILE:
    #!/bin/bash
    FILEBASE={{FILE}}
    FILEBASE=${FILEBASE%.*}
    FILEBASE=${FILEBASE%.llvm}
    {{llvm_prefix}}/bin/mlir-translate -mlir-to-llvmir {{FILE}} > ${FILEBASE}.ll
    # creates {{FILE}}.s
    {{llvm_prefix}}/bin/llc -O0 ${FILEBASE}.ll
    clang-14 -fuse-ld=lld -L{{build_dir}}/lib -lSigiRuntime ${FILEBASE}.s -g -o ${FILEBASE}.exe -no-pie

addNewDialect DIALECT_NAME DIALECT_NS:
    just --justfile ./dialectTemplate/justfile applyTemplate {{DIALECT_NAME}} {{DIALECT_NS}} "cinm-mlir" {{justfile_directory()}}

