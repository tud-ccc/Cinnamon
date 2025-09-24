#!/bin/bash

script_dir="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
source "$script_dir/common.sh"

# Ensure we have a Python interpreter available (prefer the repo venv).
if [[ -z "${VIRTUAL_ENV:-}" && -d "$py_venv_path" ]]; then
  # shellcheck disable=SC1091
  source "$py_venv_path/bin/activate"
fi

PYTHON_BIN="$(command -v python3 || command -v python || true)"
if [[ -z "$PYTHON_BIN" ]]; then
  error "No Python interpreter found (python3/python)"
  exit 1
fi

if [[ $setup_python_venv -eq 1 ]]; then
  python_for_install="$py_venv_path/bin/python"
  if [[ ! -x "$python_for_install" ]]; then
    error "Expected Python venv at $py_venv_path. Run setup-venv.sh first."
    exit 1
  fi
else
  python_for_install="$PYTHON_BIN"
fi

if ! "$python_for_install" -m pip --version >/dev/null 2>&1; then
  error "pip is not available for interpreter $python_for_install"
  exit 1
fi

if ! "$python_for_install" -m pip show wheel >/dev/null 2>&1; then
  status "Installing wheel into Python environment ($python_for_install)"
  verbose_cmd "$python_for_install" -m pip install wheel
fi

if [[ $checkout_and_build_torch_mlir -eq 1 ]]; then
  reconfigure_torch_mlir=0
  if [ ! -d "$torch_mlir_path" ]; then
    status "Checking out Torch-MLIR"
    git_clone_revision https://github.com/llvm/torch-mlir 389541fb9ddd33c3891650e47106f2c3b50b9322 "$torch_mlir_path"
    reconfigure_torch_mlir=1
  fi

  pushd "$torch_mlir_path" >/dev/null

  if [ -f build/CMakeCache.txt ] && ! grep -q 'CMAKE_GENERATOR:INTERNAL=Ninja' build/CMakeCache.txt; then
    status "Existing Torch-MLIR build dir is not Ninja → recreating build/"
    rm -rf build
    reconfigure_torch_mlir=1
  fi

  if [ ! -d build ] || [ ${reconfigure:-0} -eq 1 ] || [ $reconfigure_torch_mlir -eq 1 ]; then
    status "Configuring Torch-MLIR (Ninja)"
    dependency_paths=""

    if [[ $setup_python_venv -eq 1 ]]; then
      dependency_paths="$dependency_paths -DPython3_FIND_VIRTUALENV=ONLY"
    fi

    if [[ $checkout_and_build_llvm -eq 1 ]]; then
      dependency_paths="$dependency_paths -DLLVM_DIR=$llvm_path/build/lib/cmake/llvm"
      dependency_paths="$dependency_paths -DMLIR_DIR=$llvm_path/build/lib/cmake/mlir"
    fi

    LLVM_LIB_DIR="$llvm_path/build/lib"
    [ -d "$LLVM_LIB_DIR" ] || LLVM_LIB_DIR="$llvm_path/install/lib"

    case "$(uname -s)" in
      Darwin) LINKER_FLAGS="-L${LLVM_LIB_DIR} -Wl,-rpath,${LLVM_LIB_DIR} -lMLIRParser" ;;
      *)      LINKER_FLAGS="-Wl,--no-as-needed -L${LLVM_LIB_DIR} -Wl,-rpath,${LLVM_LIB_DIR} -lMLIRParser" ;;
    esac

    cmake -S . -B build -G Ninja \
      $dependency_paths \
      -Wno-dev \
      -DCMAKE_BUILD_TYPE=Release \
      -DTORCH_MLIR_OUT_OF_TREE_BUILD=ON \
      -DTORCH_MLIR_ENABLE_STABLEHLO=OFF \
      -U CMAKE_EXE_LINKER_FLAGS -U CMAKE_SHARED_LINKER_FLAGS \
      "-DCMAKE_EXE_LINKER_FLAGS:STRING=${LINKER_FLAGS}" \
      "-DCMAKE_SHARED_LINKER_FLAGS:STRING=${LINKER_FLAGS}" \
      "-DCMAKE_BUILD_RPATH:STRING=${LLVM_LIB_DIR}" \
      "-DCMAKE_INSTALL_RPATH:STRING=${LLVM_LIB_DIR}" \
      $TORCH_MLIR_CMAKE_OPTIONS
  fi

  status "Building Torch-MLIR (Ninja)"
  cmake --build build --target all TorchMLIRPythonModules

  verbose_cmd cmake --install build --prefix install

  if [[ $setup_python_venv -eq 1 ]]; then
    status "Building and installing Torch-MLIR Python package into $py_venv_path"
    python_package_dir=build/tools/torch-mlir/python_packages/torch_mlir
    python_package_rel_build_dir=../../../python_packages/torch_mlir
    mkdir -p "$(dirname "$python_package_dir")"
    ln -s "$python_package_rel_build_dir" "$python_package_dir" 2> /dev/null || true
    TORCH_MLIR_CMAKE_ALREADY_BUILT=1 TORCH_MLIR_CMAKE_BUILD_DIR=build PYTHONWARNINGS=ignore \
      verbose_cmd "$python_for_install" -m pip install --no-build-isolation --no-deps --force-reinstall .
  elif [[ $setup_python_venv -eq 0 ]]; then
    warning "Building Torch-MLIR Python package with interpreter: $python_for_install"
    TORCH_MLIR_CMAKE_ALREADY_BUILT=1 TORCH_MLIR_CMAKE_BUILD_DIR=build PYTHONWARNINGS=ignore \
      verbose_cmd "$python_for_install" -m pip install --no-build-isolation --no-deps --force-reinstall .
  fi

  popd >/dev/null

elif [[ $checkout_and_build_torch_mlir -eq 0 ]]; then
  warning "Skipping Torch-MLIR checkout and build"
  warning "The following steps will need TORCH_MLIR_DIR to be set in their respective <STEP>_CMAKE_OPTIONS"
fi
