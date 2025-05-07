#!/bin/bash

if [ -z $PREAMBLE_LOADED ]; then
PREAMBLE_LOADED="1"

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


project_root="$( cd -- "$(dirname "$0")/../.." >/dev/null 2>&1 ; pwd -P )"
status "Project root: $project_root"
mkdir -p "$project_root/third-party"

py_venv_path="$project_root/.venv"
cinnamon_path="$project_root"
llvm_path="$project_root/third-party/llvm"
torch_mlir_path="$project_root/third-party/torch-mlir"
upmem_path="$project_root/third-party/upmem"


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


if [ -n "$TORCH_MLIR_INSTALL_DIR" ] && [ "$checkout_and_build_torch_mlir" -ne 0 ]; then
  checkout_and_build_torch_mlir=external
  CINNAMON_CMAKE_OPTIONS="$CINNAMON_CMAKE_OPTIONS -DTORCH_MLIR_DIR=$TORCH_MLIR_INSTALL_DIR"

  info "Using environment variable TORCH_MLIR_INSTALL_DIR for configuration"
  info "Dependent targets will use '$TORCH_MLIR_INSTALL_DIR'"

  if [ ! -d "$TORCH_MLIR_INSTALL_DIR" ]; then
    warning "Directory '$TORCH_MLIR_INSTALL_DIR' does not exist"
  fi
fi

fi
