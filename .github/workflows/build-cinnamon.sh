#!/bin/bash

script_dir="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
source "$script_dir/common.sh"

cd "$cinnamon_path"

if [ ! -d "build" ] || [ $reconfigure -eq 1 ]; then
  status "Configuring Cinnamon"
  ln -s "$project_root/LICENSE" "$cinnamon_path/python/" 2>/dev/null || true

  dependency_paths=""
  
  if [[ $checkout_and_build_llvm -eq 1 ]]; then
    dependency_paths="$dependency_paths -DLLVM_DIR=$llvm_path/build/lib/cmake/llvm"
    dependency_paths="$dependency_paths -DMLIR_DIR=$llvm_path/build/lib/cmake/mlir"
  fi

  if [[ $checkout_and_build_torch_mlir -eq 1 ]]; then
    dependency_paths="$dependency_paths -DTORCH_MLIR_DIR=$torch_mlir_path/install"  
  fi

  if [[ $checkout_upmem -eq 1 ]]; then
    dependency_paths="$dependency_paths -DUPMEM_DIR=$upmem_path"
  fi

  cmake -S . -B "build" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    $dependency_paths \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    $CINNAMON_CMAKE_OPTIONS
fi

status "Building Cinnamon"
cmake --build build --target all

if [[ $setup_python_venv -eq 1 ]] && [[ -n "$llvm_path" ]] && [[ -n "$torch_mlir_path" ]]; then
  status "Building Cinnamon Python package"
  site_packages_dir="$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')"
  cinnamon_python_package_dir_src="$project_root/python/src/cinnamon"
  cinnamon_python_package_dir_dest="$site_packages_dir"
  cinnamon_python_package_resource_dir="$cinnamon_python_package_dir_dest/_resources"

  cinnamon_python_resources=""

  cinnamon_python_resources="$cinnamon_python_resources $cinnamon_path/build/bin/cinm-opt"
  cinnamon_python_resources="$cinnamon_python_resources $cinnamon_path/build/lib/libMemristorDialectRuntime.so"

  cinnamon_python_resources="$cinnamon_python_resources $torch_mlir_path/build/bin/torch-mlir-opt"

  cinnamon_python_resources="$cinnamon_python_resources $llvm_path/build/bin/mlir-translate"
  cinnamon_python_resources="$cinnamon_python_resources $llvm_path/build/bin/clang"

  if [ ! -d "$cinnamon_python_package_dir_dest" ]; then
      ln -s "$cinnamon_python_package_dir_src" "$cinnamon_python_package_dir_dest"
  fi

  mkdir -p "$cinnamon_python_package_resource_dir" || true

  for resource in $cinnamon_python_resources; do
    ln -s "$resource" "$cinnamon_python_package_resource_dir" 2>/dev/null || true
  done

  if [[ $build_cinnamon_wheel -eq 1 ]]; then
    cd "$cinnamon_path/python"
    PYTHONWARNINGS=ignore verbose_cmd python -m build
  fi
elif [[ $setup_python_venv -eq 0 ]]; then
  warning "Skipping Cinnamon Python package build"
  warning "Make sure to have a correct Python environment set up"
fi
