# Build recipes for this project.
#

# Load environment vars from .env file
# Write LLVM_BUILD_DIR="path" into that file or set this env var in your shell.
set dotenv-load := true

# Make sure your LLVM is tags/llvmorg-18.1.6

llvm_prefix := env_var_or_default("LLVM_BUILD_DIR", "")
build_type := env_var_or_default("LLVM_BUILD_TYPE", "RelWithDebInfo")
linker := env_var_or_default("CMAKE_LINKER_TYPE", "DEFAULT")
upmem_dir := env_var_or_default("UPMEM_HOME", "")
build_dir := "cinnamon/build"

# Do a full build as if in CI. Only needed the first time you build the project.
# Parameters: no-upmem enable-cuda enable-roc no-torch-mlir no-python-venv
configure *ARGS:
    .github/workflows/build-ci.sh reconfigure {{ARGS}}


# execute a specific ninja target
doNinja *ARGS:
    ninja -C{{build_dir}} {{ARGS}}


# run build --first build needs cmake though
build: doNinja

cleanBuild:
    rm -rf {{build_dir}}
    just cmake
    just build

alias b := build

# run tests
test: (doNinja "check-cinm-mlir")

[no-cd]
cinm-opt *ARGS: (doNinja "cinm-opt")
    {{source_directory()}}/{{build_dir}}/bin/cinm-opt {{ARGS}}

cinm-opt-help: (cinm-opt "--help")

# Start a gdb session on cinm-opt.
debug-cinm-opt *ARGS:
    gdb --args {{build_dir}}/bin/cinm-opt {{ARGS}}

cinm-to-cnm FILE *ARGS: (
	cinm-opt FILE
	"--cinm-tiling"
	"--affine-loop-unroll='unroll-full unroll-full-threshold=1'"
	"--convert-cinm-to-cnm"
	"--lower-affine"
	"--one-shot-bufferize='bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map'"
	"--convert-linalg-to-affine-loops"
	"--lower-affine"
	"--buffer-loop-hoisting"
	"--buffer-hoisting"
	"--cse"
	ARGS
)

cnm-to-gpu FILE *ARGS: (cinm-opt FILE "--convert-cnm-to-gpu" ARGS)
cinm-to-gpu FILE *ARGS: (cinm-to-cnm FILE "--convert-cnm-to-gpu" ARGS)

cinm-vulkan-runner FILE *ARGS:
    {{build_dir}}/bin/cinm-vulkan-runner {{FILE}} \
        --shared-libs={{llvm_prefix}}/lib/libvulkan-runtime-wrappers.so,{{llvm_prefix}}/lib/libmlir_runner_utils.so \
        --entry-point-result=void \
        {{ARGS}}

genBench NAME: (doNinja "cinm-opt")
    #!/bin/bash
    source "{{upmem_dir}}/upmem_env.sh"
    cd testbench
    export BENCH_NAME="{{NAME}}"
    make clean && make {{NAME}}-exe

runBench NAME:
    #!/bin/bash
    source "{{upmem_dir}}/upmem_env.sh"
    cd testbench/generated2/{{NAME}}/bin
    ./host

bench NAME: (doNinja "cinm-opt")
    #!/bin/bash
    set -e
    source "{{upmem_dir}}/upmem_env.sh"
    cd testbench
    export BENCH_NAME="{{NAME}}"
    make clean && make {{NAME}}-exe
    cd generated2/{{NAME}}/bin
    ./host



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
