#!/bin/bash

project_root="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
llvm_path="$project_root/llvm"
cinnamon_path="$project_root/cinnamon"
# Upmem is installed from the debian package.
upmem_path="/usr"

export PATH=$llvm_path/build/bin:$PATH

if [[ $1 != "no-llvm" ]]; then
  if [ ! -d "$llvm_path" ]; then
    git clone https://github.com/oowekyala/llvm-project "$llvm_path"

    cd "$llvm_path"

    git checkout cinnamon-llvm
    cmake -S llvm -B build \
      -DLLVM_ENABLE_PROJECTS="mlir;llvm;clang" \
      -DLLVM_TARGETS_TO_BUILD="host" \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
      -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=SPIRV \
      -DLLVM_BUILD_TOOLS=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_LINKER_TYPE=MOLD \
      -DCMAKE_C_USING_LINKER_mold=-fuse-ld=mold \
      -DCMAKE_CXX_USING_LINKER_mold=-fuse-ld=mold \
      -DLLVM_OPTIMIZED_TABLEGEN=ON
  fi

  cd "$llvm_path"
  git pull
  cmake --build build --target all llc opt
fi

cd "$cinnamon_path"

if [ ! -d "build" ]; then
  cmake -S . -B "build" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DLLVM_DIR="$llvm_path"/build/lib/cmake/llvm \
    -DMLIR_DIR="$llvm_path"/build/lib/cmake/mlir \
    -DUPMEM_DIR="$upmem_path" \
    -DCINM_BUILD_GPU_SUPPORT=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    $CINNAMON_CMAKE_OPTIONS
fi

cmake --build build --target all