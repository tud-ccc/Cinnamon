# Build recipes for this project.
#

# Load environment vars from .env file
# Write LLVM_BUILD_DIR="path" into that file or set this env var in your shell.
set dotenv-load := true
# Allows running eg `just experiments run gemv`
mod experiments

# LLVM comes from the third-party/llvm submodule
# (https://github.com/tud-ccc/cinnamon-llvm, branch `cinnamon`). To build
# against an LLVM you already have, set LLVM_BUILD_DIR; the submodule then
# stays uninitialized. See the README.

upmem_dir := env_var_or_default("UPMEM_HOME", "third-party/upmem")
build_dir := "build"

# Full build: venv, then LLVM, Torch-MLIR and Cinnamon. Only needed the first
# time; use `just build` afterwards.
# Flags: -no-llvm -no-torch-mlir -no-upmem -no-python-venv -no-cinnamon-wheel
#        -enable-gpu -enable-cuda -enable-roc -reconfigure -verbose
configure *ARGS:
    .github/workflows/build-local.sh -reconfigure {{ARGS}}


# execute a specific ninja target
doNinja *ARGS:
    ninja -C{{build_dir}} {{ARGS}}


@highlight:
    pygmentize -l docs/MlirLexer.py:MlirLexer -x -O style=github-dark /dev/stdin

# Run cinm-opt with the given arguments. You can use this if you haven't updated your PATH.
[no-cd]
cinm-opt *ARGS: (doNinja "cinm-opt")
    #!/bin/sh
    if [ -t 1 ] ; then
     {{source_directory()}}/{{build_dir}}/bin/cinm-opt {{ARGS}} | just highlight
    else
     {{source_directory()}}/{{build_dir}}/bin/cinm-opt {{ARGS}}
    fi

[no-cd]
cinm-translate *ARGS: (doNinja "cinm-translate")
    {{source_directory()}}/{{build_dir}}/bin/cinm-translate {{ARGS}}

# Rebuild LLVM. Only needed after the third-party/llvm submodule moves.
buildLlvm:
    .github/workflows/build-llvm.sh

# Incremental build of Cinnamon itself.
build: doNinja

cleanBuild:
    rm -rf {{build_dir}}
    just configure

alias b := build


# run all tests
test: (doNinja "check-cinm-mlir" "check-unit-tests")
# Run only c++ unit tests
testUnit: (doNinja "check-unit-tests")

# Run all LIT tests that match a partial substring.
testRun PAT:
  #!/bin/bash
  find build/test -iname '*{{PAT}}*' -exec bash \{\} \;



genBench NAME: (doNinja "cinm-opt" "cinm-translate")
    #!/bin/bash
    source "{{upmem_dir}}/upmem_env.sh"
    export BENCH_NAME="{{NAME}}"
    make -Ctestbench clean && make -Ctestbench {{NAME}}-exe

runBench NAME:
    #!/bin/bash
    source "{{upmem_dir}}/upmem_env.sh"
    cd testbench/gen/{{NAME}}/bin
    ./host

bench NAME: (doNinja "cinm-opt")
    #!/bin/bash
    set -e
    source "{{upmem_dir}}/upmem_env.sh"
    export BENCH_NAME="{{NAME}}"
    make -Ctestbench clean && make -Ctestbench {{NAME}}-exe
    cd testbench/gen/{{NAME}}/bin
    ./host
