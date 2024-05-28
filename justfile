# Build recipes for this project.
#

# Load environment vars from .env file
# Write LLVM_BUILD_DIR="path" into that file or set this env var in your shell.
set dotenv-load := true

llvm_prefix := env_var("LLVM_BUILD_DIR")
build_type := env_var_or_default("LLVM_BUILD_TYPE", "RelWithDebInfo")
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
        -DCMAKE_LINKER_TYPE=LLD \
        -DCMAKE_CXX_COMPILER=clang++ \

# execute a specific ninja target
doNinja *ARGS:
    ninja -C{{build_dir}} {{ARGS}}


# run build --first build needs cmake though
build: doNinja

alias b := build 

# run tests
test: (doNinja "check-cinm-mlir")

cinm-opt *ARGS:
    {{build_dir}}/bin/cinm-opt {{ARGS}}

cinm-opt-help: (cinm-opt "--help")

# Start a gdb session on cinm-opt.
debug-cinm-opt *ARGS:
    gdb --args {{build_dir}}/bin/cinm-opt {{ARGS}}

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

