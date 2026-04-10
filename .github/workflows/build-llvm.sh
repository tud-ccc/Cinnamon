#!/bin/bash

script_dir="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
source "$script_dir/common.sh"

if ! command -v ninja >/dev/null 2>&1; then
  error "Ninja not found. Install it (e.g., 'sudo apt install ninja-build')."
  exit 1
fi

if [[ ${checkout_and_build_llvm:-0} -eq 1 ]]; then
  reconfigure_llvm=0
  if [ ! -d "$llvm_path" ]; then
    status "Checking out LLVM"
    git clone https://github.com/oowekyala/llvm-project --depth 1 --branch tilefirst-llvm "$llvm_path"
    reconfigure_llvm=1
  fi

  pushd "$llvm_path" >/dev/null

  # If build/ exists but wasn’t generated with Ninja, recreate it
  if [ -f build/CMakeCache.txt ] && ! grep -q 'CMAKE_GENERATOR:INTERNAL=Ninja' build/CMakeCache.txt; then
    status "Existing LLVM build dir is not Ninja → recreating build/"
    rm -rf build
    reconfigure_llvm=1
  fi

  if [ ${reconfigure:-0} -eq 1 ] || [ $reconfigure_llvm -eq 1 ]; then
    status "Configuring LLVM (Ninja)"
    cmake -S llvm -B build -G Ninja \
      -Wno-dev \
      -DLLVM_ENABLE_PROJECTS="mlir;llvm;clang" \
      -DLLVM_TARGETS_TO_BUILD="host" \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
      -DLLVM_BUILD_TOOLS=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=ON \
      -DLLVM_INCLUDE_TESTS=OFF \
      -DLLVM_INCLUDE_BENCHMARKS=OFF \
      -DLLVM_OPTIMIZED_TABLEGEN=ON \
      -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=SPIRV \
      $LLVM_CMAKE_OPTIONS
  fi

  status "Building LLVM (Ninja)"
  # Keep your original targets, plus ensure mlir tools are built.
  cmake --build build --target all llc opt mlir-opt mlir-translate

  export PATH="$llvm_path/build/bin:$PATH"
  popd >/dev/null
else
  warning "Skipping LLVM checkout and build"
  warning "The following steps will need LLVM_DIR and MLIR_DIR to be set in their respective <STEP>_CMAKE_OPTIONS"
fi