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
# Parameters: no-upmem enable-gpu enable-cuda enable-roc no-torch-mlir no-python-venv
configure *ARGS:
    .github/workflows/build-ci.sh -reconfigure {{ARGS}}

# execute cmake -- this is only needed on the first build
cmake *ARGS:
    cmake -S . -B {{build_dir}} \
        -G Ninja \
        -DCMAKE_BUILD_TYPE={{build_type}} \
        "-DLLVM_DIR={{llvm_prefix}}/lib/cmake/llvm" \
        "-DMLIR_DIR={{llvm_prefix}}/lib/cmake/mlir" \
        "-DUPMEM_DIR={{upmem_dir}}" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_LINKER_TYPE={{linker}} \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_C_USING_LINKER_mold=-fuse-ld=mold \
        -DCMAKE_CXX_USING_LINKER_mold=-fuse-ld=mold \
        {{ARGS}}

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

[no-cd]
cinm-translate *ARGS: (doNinja "cinm-opt")
    {{source_directory()}}/{{build_dir}}/bin/cinm-translate {{ARGS}}

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
    "--one-shot-bufferize='allow-return-allocs-from-loops allow-unknown-ops copy-before-write bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map'"
    "--convert-linalg-to-affine-loops"
    "--lower-affine"
    "--buffer-loop-hoisting"
    "--buffer-hoisting"
    "--canonicalize"
    "--cse"
    ARGS
)

cnm-to-gpu FILE *ARGS: (cinm-opt FILE "--convert-cnm-to-gpu" ARGS)
cinm-to-gpu FILE *ARGS: (cinm-to-cnm FILE "--convert-cnm-to-gpu" ARGS)

cnm-to-upmem FILE *ARGS: (
    cinm-opt FILE
    "--convert-cnm-to-upmem"
    "--cse"
    "--convert-math-to-llvm"
    "--upmem-outline-kernel"
    "--upmem-dedup-kernels"
    "--cse"
    ARGS
)

upmem-to-llvm FILE *ARGS: (
    cinm-opt FILE
    "--mlir-print-debuginfo"
    "--convert-scf-to-cf"
    "--convert-cf-to-llvm"
    "--fold-memref-alias-ops"
    "--lower-affine"
    "--convert-arith-to-llvm"
    "--convert-upmem-to-llvm"
    "--expand-strided-metadata"
    "--memref-expand"
    "--finalize-memref-to-llvm"
    "--lower-affine"
    "--convert-arith-to-llvm"
    "--convert-func-to-llvm=use-bare-ptr-memref-call-conv=true"
    "--cse"
    "--reconcile-unrealized-casts"
    "--llvm-legalize-for-export"
    "--canonicalize"
    "--cse"
    ARGS
)

translate-mlir-to-llvmir FILE *ARGS: (
    cinm-translate FILE
    "--debugify-level=locations"
    "--profile-correlate=debug-info"
    "--mlir-to-llvmir"
    ARGS
)

translate-upmem-kernel-to-cpp FILE *ARGS: (
    cinm-translate FILE
    "--mlir-to-upmem-cpp"
    ARGS
)

compile-upmem-kernels FILE OUTDIR:
    cinnamon/testbench/lib/compile_dpu.sh {{FILE}} {{OUTDIR}}

compile-upmem-runner *ARGS:
    clang++ -g -c {{ARGS}}

link-upmem-runner *ARGS:
    clang++ -g {{ARGS}} -lUpmemDialectRuntime -fPIE -ldpu -ldpuverbose -L{{upmem_dir}}/lib -L{{build_dir}}/lib -I{{upmem_dir}}/include/dpu

remove-memref-alignment FILE:
	sed -i 's/{alignment = 64 : i64} //' {{FILE}}

build-transformer:
    mkdir -p {{build_dir}}/samples
    just cinm-to-cnm cinnamon/samples/transformer.mlir -o {{build_dir}}/samples/transformer.cnm.mlir
    just build-transformer-from-cnm {{build_dir}}/samples/transformer.cnm.mlir

build-transformer-from-cnm FILE:
    just cnm-to-upmem {{FILE}} -o {{build_dir}}/samples/transformer.upmem.mlir
    just remove-memref-alignment {{build_dir}}/samples/transformer.upmem.mlir
    just upmem-to-llvm {{build_dir}}/samples/transformer.upmem.mlir -o {{build_dir}}/samples/transformer.llvm.mlir
    just translate-mlir-to-llvmir {{build_dir}}/samples/transformer.llvm.mlir -o {{build_dir}}/samples/transformer.ll
    just translate-upmem-kernel-to-cpp {{build_dir}}/samples/transformer.upmem.mlir -o {{build_dir}}/samples/transformer.upmem.c
    just compile-upmem-kernels {{build_dir}}/samples/transformer.upmem.c {{build_dir}}/samples
    just compile-upmem-runner {{build_dir}}/samples/transformer.ll -o {{build_dir}}/samples/transformer.o
    just compile-upmem-runner cinnamon/samples/llama2.cpp -o {{build_dir}}/samples/llama2.o
    just link-upmem-runner {{build_dir}}/samples/transformer.o {{build_dir}}/samples/llama2.o -o {{build_dir}}/samples/transformer

cinm-vulkan-runner FILE *ARGS:
    {{build_dir}}/bin/cinm-vulkan-runner {{FILE}} \
        --shared-libs={{llvm_prefix}}/lib/libvulkan-runtime-wrappers.so,{{llvm_prefix}}/lib/libmlir_runner_utils.so \
        --entry-point-result=void \
        {{ARGS}}

genBench NAME: (doNinja "cinm-opt")
    #!/bin/bash
    source "{{upmem_dir}}/upmem_env.sh"
    cd cinnamon/testbench
    export BENCH_NAME="{{NAME}}"
    make clean && make {{NAME}}-exe

runBench NAME:
    #!/bin/bash
    source "{{upmem_dir}}/upmem_env.sh"
    cd cinnamon/testbench/generated2/{{NAME}}/bin
    ./host

bench NAME: (doNinja "cinm-opt")
    #!/bin/bash
    set -e
    source "{{upmem_dir}}/upmem_env.sh"
    cd cinnamon/testbench
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
