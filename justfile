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
python310_dir := env_var("PYTHON_310_DIR")

# Do a full build as if in CI. Only needed the first time you build the project.
# Parameters: no-upmem enable-gpu enable-cuda enable-roc no-torch-mlir no-python-venv
configure *ARGS:
    .github/workflows/build-ci.sh -reconfigure {{ARGS}}


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
    "--pass-pipeline='builtin.module(func.func(cinm-tiling,affine-loop-unroll{unroll-full unroll-full-threshold=1},convert-cinm-to-cnm,lower-affine),
    one-shot-bufferize{allow-return-allocs-from-loops allow-unknown-ops copy-before-write bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map},
    convert-linalg-to-affine-loops, lower-affine, func.func(buffer-loop-hoisting, buffer-hoisting), canonicalize, cse)'"
    ARGS
)

cnm-to-gpu FILE *ARGS: (cinm-opt FILE "--convert-cnm-to-gpu" ARGS)
cinm-to-gpu FILE *ARGS: (cinm-to-cnm FILE "--convert-cnm-to-gpu" ARGS)

cnm-to-upmem FILE *ARGS: (
    cinm-opt FILE
    "--convert-cnm-to-upmem"
    "--cse"
    "--upmem-outline-kernel"
    "--upmem-dedup-kernels"
    "--cse"
    ARGS
)

upmem-to-llvm FILE *ARGS: (
    cinm-opt FILE
    "--mlir-print-debuginfo"
    "--convert-math-to-llvm"
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
    "--mlir-to-llvmir"
    ARGS
)

translate-upmem-kernel-to-cpp FILE *ARGS: (
    cinm-translate FILE
    "--mlir-to-upmem-cpp"
    ARGS
)

compile-upmem-kernels FILE OUTDIR:
    bash "cinnamon/testbench/lib/compile_dpu.sh" {{FILE}} {{OUTDIR}}

compile-upmem-runner *ARGS:
    llvm/build/bin/clang++ -g -c {{ARGS}}

link-upmem-runner *ARGS:
    /usr/bin/clang++ -g {{ARGS}} -lUpmemDialectRuntime -fPIE -ldpu -ldpuverbose -L{{upmem_dir}}/lib -L{{build_dir}}/lib -I{{upmem_dir}}/include/dpu -rpath {{python310_dir}}

remove-memref-alignment FILE:
	sed -i 's/{alignment = 64 : i64} //' {{FILE}}

build-transformer: \
    (cinm-to-cnm "cinnamon/samples/transformer.mlir" "-o" "./transformer.cnm.mlir") \
    (build-transformer-from-cnm "./transformer.cnm.mlir")

build-transformer-from-cnm FILE: \
    (cnm-to-upmem FILE "-o" "./transformer.upmem.mlir") \
    (remove-memref-alignment "./transformer.upmem.mlir") \
    (upmem-to-llvm "./transformer.upmem.mlir" "-o" "./transformer.llvm.mlir") \
    (translate-mlir-to-llvmir "./transformer.llvm.mlir" "-o" "./transformer.ll") \
    (translate-upmem-kernel-to-cpp "./transformer.upmem.mlir" "-o" "./transformer.upmem.c") \
    (compile-upmem-kernels "./transformer.upmem.c" "cinnamon/build/samples") \
    (compile-upmem-runner "./transformer.ll" "-o" "cinnamon/build/samples/transformer.o") \
    (compile-upmem-runner "cinnamon/samples/llama2.cpp" "-o" "cinnamon/build/samples/llama2.o") \
    (link-upmem-runner "cinnamon/build/samples/transformer.o" "cinnamon/build/samples/llama2.o" "-o" "cinnamon/build/samples/transformer")

build-dorado: \
    (cinm-to-cnm "cinnamon/samples/dorado/dorado.mlir" "-o" "./dorado.cnm.mlir") \
    (cnm-to-upmem "./dorado.cnm.mlir" "-o" "./dorado.upmem.mlir") \
    (remove-memref-alignment "./dorado.upmem.mlir") \
    (upmem-to-llvm "./dorado.upmem.mlir" "-o" "./dorado.llvm.mlir") \
    (translate-mlir-to-llvmir "./dorado.llvm.mlir" "-o" "./dorado.ll") \
    (translate-upmem-kernel-to-cpp "./dorado.upmem.mlir" "-o" "./dorado.upmem.c") \
    (compile-upmem-kernels "./dorado.upmem.c" "cinnamon/build/samples") \
    (compile-upmem-runner "./dorado.ll" "-o" "cinnamon/build/samples/dorado.o") \
    (compile-upmem-runner "cinnamon/samples/dorado/dorado.cpp" "-o" "cinnamon/build/samples/dorado_host.o" "-fopenmp") \
    (link-upmem-runner "cinnamon/build/samples/dorado.o" "cinnamon/build/samples/dorado_host.o" "-o" "cinnamon/build/samples/dorado" "-fopenmp")

[working-directory: 'cinnamon/build/samples']
dorado:
    ./dorado ~/Cinnamon/test

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
