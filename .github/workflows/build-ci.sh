#!/bin/bash

set -e

function status {
  echo -e "\033[1m$1\033[0m"
}

function info {
  echo -e "\033[1;34m$1\033[0m"
}

function warning {
  echo -e "\033[1;33m$1\033[0m"
}

function verbose_cmd {
  if [ $verbose -eq 1 ]; then
    "$@"
  else
    "$@" > /dev/null
  fi
}

project_root="$( cd -- "$(dirname "$0")/../.." >/dev/null 2>&1 ; pwd -P )"
status "Project root: $project_root"

py_venv_path="$project_root/.venv"
cinnamon_path="$project_root/cinnamon"
llvm_path="$project_root/llvm"
torch_mlir_path="$project_root/torch-mlir"
upmem_path="$project_root/upmem"

verbose=0
reconfigure=0

setup_python_venv=1
checkout_and_build_llvm=1
checkout_and_build_torch_mlir=1
checkout_upmem=1

build_cinnamon_wheel=1

enable_cuda=0
enable_roc=0

# Section for configuring based on legacy environment variables
###############################################################

if [ -n "$LLVM_BUILD_DIR" ]; then
  checkout_and_build_llvm=external
  TORCH_MLIR_CMAKE_OPTIONS="$TORCH_MLIR_CMAKE_OPTIONS -DLLVM_DIR=$LLVM_BUILD_DIR/lib/cmake/llvm -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir"
  CINNAMON_CMAKE_OPTIONS="$CINNAMON_CMAKE_OPTIONS -DLLVM_DIR=$LLVM_BUILD_DIR/lib/cmake/llvm -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir"

  info "Using environment variable LLVM_BUILD_DIR for configuration"
  info "Dependent targets will use '$LLVM_BUILD_DIR'"

  if [ ! -d "$LLVM_BUILD_DIR" ]; then
    warning "Directory '$LLVM_BUILD_DIR' does not exist"
  fi
fi

if [ -n "$TORCH_MLIR_INSTALL_DIR" ]; then
  checkout_and_build_torch_mlir=external
  CINNAMON_CMAKE_OPTIONS="$CINNAMON_CMAKE_OPTIONS -DTORCH_MLIR_DIR=$TORCH_MLIR_INSTALL_DIR"

  info "Using environment variable TORCH_MLIR_INSTALL_DIR for configuration"
  info "Dependent targets will use '$TORCH_MLIR_INSTALL_DIR'"

  if [ ! -d "$TORCH_MLIR_INSTALL_DIR" ]; then
    warning "Directory '$TORCH_MLIR_INSTALL_DIR' does not exist"
  fi
fi

if [ -n "$UPMEM_HOME" ]; then
  checkout_upmem=external
  CINNAMON_CMAKE_OPTIONS="$CINNAMON_CMAKE_OPTIONS -DUPMEM_DIR=$UPMEM_HOME"

  info "Using environment variable UPMEM_HOME for configuration"
  info "Dependent targets will use '$UPMEM_HOME'"

  if [ ! -d "$UPMEM_HOME" ]; then
    warning "Directory '$UPMEM_HOME' does not exist"
  fi
fi

###############################################################

if echo "$@" | grep -q -- "-verbose"; then
  verbose=1
else
  info "Some steps will be run in quiet mode, use -verbose to see all output"
fi

if echo "$@" | grep -q -- "-reconfigure"; then
  reconfigure=1
fi

if echo "$@" | grep -q -- "-no-python-venv"; then
  setup_python_venv=0
fi

if echo "$@" | grep -q -- "-no-llvm"; then
  checkout_and_build_llvm=0
fi

if echo "$@" | grep -q -- "-no-torch-mlir"; then
  checkout_and_build_torch_mlir=0
fi

if echo "$@" | grep -q -- "-no-upmem"; then
  checkout_upmem=0
fi

if echo "$@" | grep -q -- "-no-cinnamon-wheel"; then
  build_cinnamon_wheel=0
fi

if echo "$@" | grep -q -- "-enable-gpu"; then
  CINNAMON_CMAKE_OPTIONS="$CINNAMON_CMAKE_OPTIONS -DCINM_BUILD_GPU_SUPPORT=ON"
fi

if echo "$@" | grep -q -- "-enable-cuda"; then
  enable_cuda=1
fi

if echo "$@" | grep -q -- "-enable-roc"; then
  enable_roc=1
fi

function git_clone_revision() {
  repo_url=$1
  revision=$2
  path=$3
  
  if (echo a version 2.49.0; git --version) | sort -Vk3 | tail -1 | grep -q git; then
    # Git 2.49.0 added the revision option
    git clone --revision "$revision" --depth 1 "$repo_url" "$path"
  else
    mkdir -p "$path"
    pushd "$path"
    git init
    git remote add origin "$repo_url"
    git fetch origin "$revision"
    git checkout FETCH_HEAD
    popd
  fi
}

if [[ $setup_python_venv -eq 1 ]]; then
  # NOTE: This is a temporary workaround as some distros ship python3.13 which does not yet provide a torch package
  supported_python_executable=python3
  if command -v python3.12 &> /dev/null; then
    supported_python_executable=python3.12
  fi

  reconfigure_python_venv=0
  if [ ! -d "$py_venv_path" ]; then
    status "Creating Python venv"
    
    if ! $supported_python_executable -m venv "$py_venv_path"; then
      echo "Error: cannot create venv"
      exit 1
    fi
    source "$py_venv_path/bin/activate"
    reconfigure_python_venv=1
  else
    status "Enabling Python venv"
    source "$py_venv_path/bin/activate"
  fi

  if [ $reconfigure -eq 1 ] || [ $reconfigure_python_venv -eq 1 ]; then
    status "Installing Python dependencies"
    # https://pytorch.org/get-started/locally/
    if [[ $enable_cuda -eq 1 ]]; then
      torch_source=https://download.pytorch.org/whl/cu124
    elif [[ $enable_roc -eq 1 ]]; then
      torch_source=https://download.pytorch.org/whl/rocm6.1
    else
      torch_source=https://download.pytorch.org/whl/cpu
    fi

    verbose_cmd pip install --upgrade pip
    verbose_cmd pip install torch torchvision torchaudio --index-url $torch_source
    verbose_cmd pip install pybind11 nanobind build numpy 
  fi
elif [[ $setup_python_venv -eq 0 ]]; then
  warning "Skipping Python venv setup"
  warning "Make sure to have a correct Python environment set up"
fi

if [[ $checkout_and_build_llvm -eq 1 ]]; then
  reconfigure_llvm=0
  if [ ! -d "$llvm_path" ]; then
    status "Checking out LLVM"
    git_clone_revision  https://github.com/llvm/llvm-project 8885b5c0626065274cb8f8a634d45779a0f6ff2b "$llvm_path"

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

if [[ $checkout_and_build_torch_mlir -eq 1 ]]; then
  reconfigure_torch_mlir=0
  if [ ! -d "$torch_mlir_path" ]; then
    status "Checking out Torch-MLIR"
    git_clone_revision  https://github.com/llvm/torch-mlir 0c29ccf1439c91c7a2175a167d4bdb2c01a03e63 "$torch_mlir_path"
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
    mkdir -p $(dirname $python_package_dir)
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

if [[ $checkout_upmem -eq 1 ]]; then
  if [ ! -d "$upmem_path" ]; then
    status "Downloading UpMem SDK"
    upmem_archive="upmem.tar.gz"
    curl http://sdk-releases.upmem.com/2024.1.0/ubuntu_22.04/upmem-2024.1.0-Linux-x86_64.tar.gz --output "$upmem_archive"
    mkdir "$upmem_path"
    tar xf "$upmem_archive" -C "$upmem_path" --strip-components=1
    rm "$upmem_archive"
  fi
elif [[ $checkout_upmem -eq 0 ]]; then
  warning "Skipping UpMem checkout"
  warning "The following steps will need UPMEM_DIR to be set in their respective <STEP>_CMAKE_OPTIONS"
fi

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
  cinnamon_python_package_dir_src="$project_root/cinnamon/python/src/cinnamon"
  cinnamon_python_package_dir_dest="$site_packages_dir/cinnamon"
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
