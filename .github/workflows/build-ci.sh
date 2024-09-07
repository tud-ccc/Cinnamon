#!/bin/bash

project_root="$( cd -- "$(dirname "$0")/../.." >/dev/null 2>&1 ; pwd -P )"
echo "Project root: $project_root"

py_venv_path="$project_root/.venv"
cinnamon_path="$project_root/cinnamon"
llvm_path="$project_root/llvm"
torch_mlir_path="$project_root/torch-mlir"
upmem_path="$project_root/upmem"

setup_python_venv=true
checkout_and_build_llvm=true
checkout_and_build_torch_mlir=true
checkout_upmem=true

if echo "$@" | grep -q "no-python-venv"; then
  setup_python_venv=false
fi

if echo "$@" | grep -q "no-llvm"; then
  checkout_and_build_llvm=false
fi

if echo "$@" | grep -q "no-torch-mlir"; then
  checkout_and_build_torch_mlir=false
fi

if echo "$@" | grep -q "no-upmem"; then
  checkout_upmem=false
fi

if [[ $setup_python_venv ]]; then
  deactivate 2> /dev/null

  if [ ! -d "$py_venv_path" ]; then
    python3 -m venv "$py_venv_path"
    source "$py_venv_path/bin/activate"
    pip install --upgrade pip
    # https://pytorch.org/get-started/locally/
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install numpy pybind11
  else
    source "$py_venv_path/bin/activate"
  fi
else
  echo "Skipping Python venv setup"
  echo "Make sure to have a correct Python environment set up"
fi

if [[ checkout_and_build_llvm ]]; then
  if [ ! -d "$llvm_path" ]; then
    git clone https://github.com/oowekyala/llvm-project "$llvm_path"

    cd "$llvm_path"

    git checkout cinnamon-llvm
    cmake -S llvm -B build \
      -DLLVM_ENABLE_PROJECTS="mlir;llvm;clang" \
      -DLLVM_TARGETS_TO_BUILD="host" \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
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

  export PATH=$llvm_path/build/bin:$PATH
else
  echo "Skipping LLVM checkout and build"
  echo "The following steps will need LLVM_DIR and MLIR_DIR to be set in their respective <STEP>_CMAKE_OPTIONS"
fi

if [[ checkout_and_build_torch_mlir ]]; then
  if [ ! -d "$torch_mlir_path" ]; then
    git clone https://github.com/llvm/torch-mlir "$torch_mlir_path"

    cd "$torch_mlir_path"

    dependency_paths=""

    if [[ checkout_and_build_llvm ]]; then
      dependency_paths="$dependency_paths -DLLVM_DIR=$llvm_path/build/lib/cmake/llvm"
      dependency_paths="$dependency_paths -DMLIR_DIR=$llvm_path/build/lib/cmake/mlir"
    fi

    git checkout snapshot-20240127.1096
    cmake -S . -B build \
      $dependency_paths \
      -DTORCH_MLIR_OUT_OF_TREE_BUILD=ON \
      -DTORCH_MLIR_ENABLE_STABLEHLO=OFF \
      $TORCH_MLIR_CMAKE_OPTIONS
  fi

  cd "$torch_mlir_path"
  cmake --build build
  cmake --install build --prefix install
else
  echo "Skipping Torch-MLIR checkout and build"
  echo "The following steps will need TORCH_MLIR_DIR to be set in their respective <STEP>_CMAKE_OPTIONS"
fi

if [[ checkout_upmem ]]; then
  if [ ! -d "$upmem_path" ]; then
    upmem_archive="upmem.tar.gz"
    curl http://sdk-releases.upmem.com/2024.1.0/ubuntu_22.04/upmem-2024.1.0-Linux-x86_64.tar.gz --output "$upmem_archive"
    mkdir "$upmem_path"
    tar xf "$upmem_archive" -C "$upmem_path" --strip-components=1
    rm "$upmem_archive"
  fi
else
  echo "Skipping UpMem checkout"
  echo "The following steps will need UPMEM_DIR to be set in their respective <STEP>_CMAKE_OPTIONS"
fi

cd "$cinnamon_path"

if [ ! -d "build" ]; then

  dependency_paths=""
  
  if [[ checkout_and_build_llvm ]]; then
    dependency_paths="$dependency_paths -DLLVM_DIR=$llvm_path/build/lib/cmake/llvm"
    dependency_paths="$dependency_paths -DMLIR_DIR=$llvm_path/build/lib/cmake/mlir"
  fi

  if [[ checkout_and_build_torch_mlir ]]; then
    dependency_paths="$dependency_paths -DTORCH_MLIR_DIR=$torch_mlir_path/install"  
  fi

  if [[ checkout_upmem ]]; then
    dependency_paths="$dependency_paths -DUPMEM_DIR=$upmem_path"
  fi

  cmake -S . -B "build" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    $dependency_paths \
    -DCINM_BUILD_GPU_SUPPORT=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    $CINNAMON_CMAKE_OPTIONS
fi

cmake --build build --target all