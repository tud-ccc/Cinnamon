#!/bin/bash

script_dir="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
source "$script_dir/common.sh"


if [[ $checkout_and_build_torch_mlir -eq 1 ]]; then
  reconfigure_torch_mlir=0
  if [ ! -d "$torch_mlir_path" ]; then
    status "Checking out Torch-MLIR"
    git_clone_revision  https://github.com/llvm/torch-mlir 389541fb9ddd33c3891650e47106f2c3b50b9322 "$torch_mlir_path"
    reconfigure_torch_mlir=1
  fi

  cd "$torch_mlir_path"

  if [ $reconfigure -eq 1 ] || [ $reconfigure_torch_mlir -eq 1 ]; then
    status "Configuring Torch-MLIR"
    dependency_paths=""

    if [[ $setup_python_venv -eq 1 ]]; then
      dependency_paths="$dependency_paths -DPython3_FIND_VIRTUALENV=ONLY"
    fi

    if [[ $checkout_and_build_llvm -eq 1 ]]; then
      dependency_paths="$dependency_paths -DLLVM_DIR=$llvm_path/build/lib/cmake/llvm"
      dependency_paths="$dependency_paths -DMLIR_DIR=$llvm_path/build/lib/cmake/mlir"
    fi
    
    status "TORCH_MLIR_CMAKE_OPTIONS=$TORCH_MLIR_CMAKE_OPTIONS"
    cmake -S . -B build \
      $dependency_paths \
      -Wno-dev \
      -DCMAKE_BUILD_TYPE=Release \
      -DTORCH_MLIR_OUT_OF_TREE_BUILD=ON \
      -DTORCH_MLIR_ENABLE_STABLEHLO=OFF \
      $TORCH_MLIR_CMAKE_OPTIONS
  fi

  status "Building Torch-MLIR"
  cmake --build build --target all TorchMLIRPythonModules
  verbose_cmd cmake --install build --prefix install

  if [[ $setup_python_venv -eq 1 ]]; then
    status "Building and installing Torch-MLIR Python package"
    python_package_dir=build/tools/torch-mlir/python_packages/torch_mlir
    python_package_rel_build_dir=../../../python_packages/torch_mlir
    mkdir -p "$(dirname "$python_package_dir")"
    ln -s "$python_package_rel_build_dir" "$python_package_dir" 2> /dev/null || true
    TORCH_MLIR_CMAKE_ALREADY_BUILT=1 TORCH_MLIR_CMAKE_BUILD_DIR=build PYTHONWARNINGS=ignore verbose_cmd python setup.py build install
  elif [[ $setup_python_venv -eq 0 ]]; then
    warning "Skipping Torch-MLIR Python package build"
    warning "Make sure to have a correct Python environment set up"
  fi

elif [[ $checkout_and_build_torch_mlir -eq 0 ]]; then
  warning "Skipping Torch-MLIR checkout and build"
  warning "The following steps will need TORCH_MLIR_DIR to be set in their respective <STEP>_CMAKE_OPTIONS"
fi
