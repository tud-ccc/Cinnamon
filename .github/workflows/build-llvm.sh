#!/bin/bash

script_dir="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
source "$script_dir/common.sh"


if [[ $checkout_and_build_llvm -eq 1 ]]; then
  reconfigure_llvm=0
  if [ ! -d "$llvm_path" ]; then
    status "Checking out LLVM"
    git clone  https://github.com/oowekyala/llvm-project --depth 1 --branch tilefirst-llvm "$llvm_path"

    reconfigure_llvm=1
  fi
  cd "$llvm_path"


  if [ $reconfigure -eq 1 ] || [ $reconfigure_llvm -eq 1 ]; then
    status "Configuring LLVM"
    cmake -S llvm -B build \
      -Wno-dev \
      -DLLVM_ENABLE_PROJECTS="mlir;llvm;clang" \
      -DLLVM_TARGETS_TO_BUILD="host" \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=ON \
      -DLLVM_INCLUDE_TESTS=OFF \
      -DLLVM_INCLUDE_BENCHMARKS=OFF \
      -DLLVM_OPTIMIZED_TABLEGEN=ON \
      -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=SPIRV \
      $LLVM_CMAKE_OPTIONS
  fi

  status "Building LLVM"
  cmake --build build --target all llc opt

  export PATH=$llvm_path/build/bin:$PATH
elif [[ $checkout_and_build_llvm -eq 0 ]]; then
  warning "Skipping LLVM checkout and build"
  warning "The following steps will need LLVM_DIR and MLIR_DIR to be set in their respective <STEP>_CMAKE_OPTIONS"
fi
