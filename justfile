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

# The environment `just install` installs into. pixi points VIRTUAL_ENV at the
# environment prefix and sets no CONDA_PREFIX; a conda environment does the
# reverse. The Python package resolves against this same environment, which is
# what lets it find a build without being told where the checkout is.
prefix := env_var_or_default("VIRTUAL_ENV", env_var_or_default("CONDA_PREFIX", ""))

# Under a subdirectory of its own rather than in bin/ and lib/ directly. Our
# LLVM is a whole toolchain -- ~350 shared libraries, the llvm-* and mlir-*
# tools, and 160M of headers -- and the environment is shared with conda
# packages that carry an LLVM of their own. Nothing collides today, but a
# later `llvm-tools` would land on the same opt and llc, and include/ is on
# the search path of every compile in the environment. Here, nothing of ours
# is in anyone's way. $ORIGIN/../lib resolves within the subtree, so the
# installed tools are no less relocatable for it.
install_dir := prefix + "/libexec/cinnamon"

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
    #!/usr/bin/env bash
    set -euo pipefail
    # Straight into the environment, because that is where the Python package
    # looks: an install left to a separate step would silently lag the build
    # tree, and the pipelines would price an old compiler. Incremental, and a
    # fraction of a second once the first copy is done. LLVM is the expensive
    # half and stays in `installLlvm`.
    if [ -n "{{prefix}}" ]; then
        cmake --install {{build_dir}} --prefix "{{install_dir}}" >/dev/null
    else
        echo "No environment is active, so nothing was installed." >&2
    fi

cleanBuild:
    rm -rf {{build_dir}}
    just configure

alias b := build


# Everything the environment needs to run a build from outside this checkout:
# `build` puts the tools, libraries, runtime headers and benchmark suites
# there, and `installLlvm` the shared libraries they load. This is the one
# command another repository's setup has to call.

# Install Cinnamon and the LLVM it loads into the environment
install: build installLlvm
    #!/usr/bin/env bash
    set -euo pipefail
    source .github/workflows/common.sh >/dev/null
    # torch-mlir-opt is the torch backend's, and torch-mlir has an install
    # tree of its own; only that one tool is wanted here.
    if [ -x "$torch_mlir_build_dir/bin/torch-mlir-opt" ]; then
        cp -a "$torch_mlir_build_dir/bin/torch-mlir-opt" "{{install_dir}}/bin/torch-mlir-opt"
    else
        echo "No torch-mlir-opt; the torch backend will not find it." >&2
    fi

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
    stamp="{{install_dir}}/$llvm_prebuilt_stamp"
    if [ "$(cat "$stamp" 2>/dev/null)" = "$llvm_revision" ]; then
        echo "LLVM ${llvm_revision:0:12} is already installed in {{install_dir}}"
        exit 0
    fi
    if [ -f "$llvm_build_dir/CMakeCache.txt" ]; then
        cmake --install "$llvm_build_dir" --prefix "{{install_dir}}"
    else
        # A prebuilt LLVM was unpacked as an install tree already.
        echo "Copying the prebuilt LLVM from $llvm_build_dir"
        mkdir -p "{{install_dir}}"
        cp -a "$llvm_build_dir/." "{{install_dir}}/"
    fi
    printf '%s\n' "$llvm_revision" > "$stamp"

# Neither pixi nor conda tracks what was installed, so this is the way back
# short of recreating the environment. The whole directory goes, rather than
# the files CMake's manifests name: those miss the symlinks LLVM installs for
# its tool aliases, and nothing but ours is in here to begin with.

# Remove everything `just install` put in the environment
uninstall:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -z "{{prefix}}" ]; then
        echo "No environment is active; activate one or set VIRTUAL_ENV." >&2
        exit 1
    fi
    # Refuse anything that is not the directory this project installs into,
    # since what follows is recursive.
    case "{{install_dir}}" in
        */libexec/cinnamon) ;;
        *) echo "Refusing to remove '{{install_dir}}'." >&2; exit 1 ;;
    esac
    if [ -d "{{install_dir}}" ]; then
        rm -rf "{{install_dir}}"
        echo "Removed {{install_dir}}"
    else
        echo "Nothing installed at {{install_dir}}"
    fi


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
