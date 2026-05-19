# Build recipes for this project.
#

# Load environment vars from .env file
# Write LLVM_BUILD_DIR="path" into that file or set this env var in your shell.
set dotenv-load := true
# Allows running eg `just experiments run gemv`
mod experiments

# Make sure your LLVM is https://github.com/oowekyala/llvm-project/tree/tilefirst-llvm

llvm_prefix := env_var_or_default("LLVM_BUILD_DIR", "")
build_type := env_var_or_default("LLVM_BUILD_TYPE", "RelWithDebInfo")
linker := env_var_or_default("CMAKE_LINKER_TYPE", "DEFAULT")
upmem_dir := env_var_or_default("UPMEM_HOME", "third-party/upmem")
build_dir := "build"

# Do a full build as if in CI. Only needed the first time you build the project.
# Parameters: no-upmem enable-gpu enable-cuda enable-roc no-torch-mlir no-python-venv
configure *ARGS:
    .github/workflows/build-local.sh -reconfigure {{ARGS}}


# execute a specific ninja target
doNinja *ARGS:
    ninja -C{{build_dir}} {{ARGS}}


@highlight:
    pygmentize -l docs/MlirLexer.py:MlirLexer -x -O style=github-dark /dev/stdin

# Run tilefirst-opt with the given arguments. You can use this if you haven't updated your PATH.
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

buildLlvm:
  #!/bin/sh
  cd third-party/llvm
  just build

# run build --first build needs cmake though
build: buildLlvm doNinja

cleanBuild:
    rm -rf {{build_dir}}
    just cmake
    just build

alias b := build

# run tests
test: (doNinja "check-cinm-mlir")

runTest PAT:
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
