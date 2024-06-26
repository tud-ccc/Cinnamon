#!/bin/bash

if [[ $1 != "no-llvm" ]]; then

git clone https://github.com/oowekyala/llvm-project llvm
cd llvm
git checkout cinnamon-llvm
mkdir -p build   
cd build

cmake -G "Ninja" ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir;llvm;clang" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
  -DLLVM_BUILD_TOOLS=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DLLVM_OPTIMIZED_TABLEGEN=ON

ninja
ninja llc
ninja opt

export PATH=$(pwd)/bin:$PATH

cd ../..
else
export PATH=$(pwd)/llvm/build/bin:$PATH
fi 

cd cinnamon 
llvm_prefix=../llvm/build

cmake -S . -B "build" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DLLVM_DIR="$llvm_prefix"/lib/cmake/llvm \
    -DMLIR_DIR="$llvm_prefix"/lib/cmake/mlir \
    -DUPMEM_DIR=/opt/upmem/upmem-2023.2.0-Linux-x86_64 \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_LINKER_TYPE=DEFAULT \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON 

cd build && ninja