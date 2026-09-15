# Build recipes for this project.
#

# Load environment vars from .env file
# Write LLVM_BUILD_DIR="path" into that file or set this env var in your shell.
set dotenv-load := true
# LLVM is the third-party/llvm submodule
# (https://github.com/tud-ccc/cinnamon-llvm, branch `cinnamon`), which is
# usually downloaded prebuilt rather than built. To build against an LLVM you
# already have, set LLVM_BUILD_DIR; the submodule then stays uninitialized.
# See BUILDING.md.

upmem_dir := env_var_or_default("UPMEM_HOME", "third-party/upmem")
build_dir := "build"

# Where `just install` puts everything. pixi points VIRTUAL_ENV at the
# environment prefix and sets no CONDA_PREFIX; a conda environment does the
# reverse. The Python package resolves against this same prefix, which is what
# lets it find a build without being told where the checkout is.
prefix := env_var_or_default("VIRTUAL_ENV", env_var_or_default("CONDA_PREFIX", ""))

# Full build: venv, then LLVM, Torch-MLIR and Cinnamon. Only needed the first
# time; use `just build` afterwards.
# Flags: -no-llvm -no-torch-mlir -no-upmem -no-python-venv -no-cinnamon-wheel
#        -enable-gpu -enable-cuda -enable-roc -reconfigure -verbose
configure *ARGS:
    .github/workflows/build-local.sh {{ARGS}}


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

# Incremental build of Cinnamon itself.
build: doNinja

cleanBuild:
    rm -rf {{build_dir}}
    just configure

alias b := build


# This is what makes a build usable from outside this checkout. The Python
# package resolves everything against the environment prefix, so a pipeline in
# another repository finds a compiler by having this run, rather than by being
# handed a path into a build tree.
#
# Deliberately not part of `build`: that is the inner loop, and this copies
# about a gigabyte the first time. It does depend on `build`, though -- an
# install of a stale build is worse than no install.

# Install the tools, libraries, runtime headers and benchmarks into the environment
install: build installLlvm
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -z "{{prefix}}" ]; then
        echo "No environment is active; activate one or set VIRTUAL_ENV." >&2
        exit 1
    fi
    cmake --install {{build_dir}} --prefix "{{prefix}}"

# The installed tools load some 350 MLIR shared libraries and find them
# through $ORIGIN/../lib, so they have to sit beside them. Keyed on the pinned
# revision -- the stamp build-llvm.sh writes beside a prebuilt tree -- because
# this is the expensive half and only changes when the submodule moves.

# Install LLVM and MLIR into the same prefix (~1GB, skipped once present)
installLlvm:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -z "{{prefix}}" ]; then
        echo "No environment is active; activate one or set VIRTUAL_ENV." >&2
        exit 1
    fi
    source .github/workflows/common.sh >/dev/null
    stamp="{{prefix}}/$llvm_prebuilt_stamp"
    if [ "$(cat "$stamp" 2>/dev/null)" = "$llvm_revision" ]; then
        echo "LLVM ${llvm_revision:0:12} is already installed in {{prefix}}"
        exit 0
    fi
    if [ -f "$llvm_build_dir/CMakeCache.txt" ]; then
        cmake --install "$llvm_build_dir" --prefix "{{prefix}}"
    else
        # A prebuilt LLVM was unpacked as an install tree already.
        echo "Copying the prebuilt LLVM from $llvm_build_dir"
        cp -a "$llvm_build_dir/." "{{prefix}}/"
    fi
    printf '%s\n' "$llvm_revision" > "$stamp"

# Neither pixi nor conda tracks what was installed, so this is the way back
# short of recreating the environment.

# Remove what `just install` added, by the manifests CMake wrote
uninstall:
    #!/usr/bin/env bash
    set -euo pipefail
    source .github/workflows/common.sh >/dev/null
    for manifest in "{{build_dir}}/install_manifest.txt" "$llvm_build_dir/install_manifest.txt"; do
        [ -f "$manifest" ] || continue
        echo "Removing the files listed in $manifest"
        # Only what the manifest names, and only if it is still a file. The
        # `|| [ -n "$f" ]` is for the last line: CMake writes no trailing
        # newline, and plain `read` would drop it.
        while IFS= read -r f || [ -n "$f" ]; do
            if [ -f "$f" ] || [ -L "$f" ]; then
                rm -f "$f"
            fi
        done < "$manifest"
    done
    rm -f "{{prefix}}/$llvm_prebuilt_stamp"


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
    make -Cbenchmarks/cinm1 clean && make -Cbenchmarks/cinm1 {{NAME}}-exe

runBench NAME:
    #!/bin/bash
    source "{{upmem_dir}}/upmem_env.sh"
    cd benchmarks/cinm1/gen/{{NAME}}/bin
    ./host

bench NAME: (doNinja "cinm-opt")
    #!/bin/bash
    set -e
    source "{{upmem_dir}}/upmem_env.sh"
    export BENCH_NAME="{{NAME}}"
    make -Cbenchmarks/cinm1 clean && make -Cbenchmarks/cinm1 {{NAME}}-exe
    cd benchmarks/cinm1/gen/{{NAME}}/bin
    ./host
