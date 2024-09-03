#!/bin/bash

project_root="$( cd -- "$(dirname "$0")/../.." >/dev/null 2>&1 ; pwd -P )"
echo "Project root: $project_root"

llvm_path="$project_root/llvm"
cinnamon_path="$project_root/cinnamon"
upmem_path="$project_root/upmem"

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
      -DLLVM_BUILD_TOOLS=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=ON \
      -DLLVM_OPTIMIZED_TABLEGEN=ON \
      -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=SPIRV \
      $LLVM_CMAKE_OPTIONS
  fi

  cd "$llvm_path"
  git pull
  cmake --build build --target all llc opt
fi

if [ ! -d "$upmem_path" ]; then
  upmem_archive="upmem.tar.gz"
  curl http://sdk-releases.upmem.com/2024.1.0/ubuntu_22.04/upmem-2024.1.0-Linux-x86_64.tar.gz --output "$upmem_archive"
  mkdir "$upmem_path"
  tar xf "$upmem_archive" -C "$upmem_path" --strip-components=1
  rm "$upmem_archive"
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