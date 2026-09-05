#!/usr/bin/env bash
set -euo pipefail

script_dir="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
# shellcheck source=/dev/null
source "$script_dir/common.sh"

if [[ "$build_torch_mlir" -eq 0 ]]; then
  warning "Skipping Torch-MLIR build; Cinnamon will use '$torch_mlir_install_dir'"
  exit 0
fi

if [[ $setup_python_venv -eq 1 ]]; then
  python_for_install="$py_venv_path/bin/python"
  if [[ ! -x "$python_for_install" ]]; then
    error "Expected Python venv at $py_venv_path. Run setup-venv.sh first."
    exit 1
  fi
else
  python_for_install="$(command -v python3 || command -v python || true)"
  if [[ -z "$python_for_install" ]]; then
    error "No Python interpreter found (python3/python)"
    exit 1
  fi
fi

if ! "$python_for_install" -m pip --version >/dev/null 2>&1; then
  error "pip is not available for interpreter $python_for_install"
  exit 1
fi

if ! "$python_for_install" -m pip show wheel >/dev/null 2>&1; then
  status "Installing wheel into Python environment ($python_for_install)"
  verbose_cmd "$python_for_install" -m pip install wheel
fi

# Torch-MLIR vendors its own llvm-project and stablehlo submodules. We build it
# out of tree against our LLVM and with StableHLO disabled, so neither is
# checked out: this init is deliberately not recursive.
ensure_submodule third-party/torch-mlir

cache_file="$torch_mlir_build_dir/CMakeCache.txt"
need_config=0

if [[ ! -f "$cache_file" ]]; then
  need_config=1
elif ! grep -q 'CMAKE_GENERATOR:INTERNAL=Ninja' "$cache_file"; then
  status "Existing Torch-MLIR build dir is not Ninja -> recreating it"
  rm -rf "$torch_mlir_build_dir"
  need_config=1
elif [[ "$reconfigure" -eq 1 ]]; then
  need_config=1
fi

if [[ "$need_config" -eq 1 ]]; then
  status "Configuring Torch-MLIR (Ninja)"
  dependency_paths=( -DLLVM_DIR="$llvm_cmake_dir" -DMLIR_DIR="$mlir_cmake_dir" )

  if [[ $setup_python_venv -eq 1 ]]; then
    dependency_paths+=( -DPython3_FIND_VIRTUALENV=ONLY )
  fi

  llvm_lib_dir="$llvm_build_dir/lib"
  case "$(uname -s)" in
    Darwin) linker_flags="-L${llvm_lib_dir} -Wl,-rpath,${llvm_lib_dir} -lMLIRParser" ;;
    *)      linker_flags="-Wl,--no-as-needed -L${llvm_lib_dir} -Wl,-rpath,${llvm_lib_dir} -lMLIRParser" ;;
  esac

  extra_opts=()
  if [[ -n "${TORCH_MLIR_CMAKE_OPTIONS:-}" ]]; then
    # shellcheck disable=SC2206
    extra_opts=( ${TORCH_MLIR_CMAKE_OPTIONS} )
  fi

  print_and_run cmake -S "$torch_mlir_source_dir" -B "$torch_mlir_build_dir" -G Ninja \
    "${dependency_paths[@]}" \
    -Wno-dev \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_EH=ON \
    -DLLVM_ENABLE_RTTI=ON \
    -DTORCH_MLIR_OUT_OF_TREE_BUILD=ON \
    -DTORCH_MLIR_ENABLE_STABLEHLO=OFF \
    -U CMAKE_EXE_LINKER_FLAGS -U CMAKE_SHARED_LINKER_FLAGS \
    "-DCMAKE_EXE_LINKER_FLAGS:STRING=${linker_flags}" \
    "-DCMAKE_SHARED_LINKER_FLAGS:STRING=${linker_flags}" \
    "-DCMAKE_BUILD_RPATH:STRING=${llvm_lib_dir}" \
    "-DCMAKE_INSTALL_RPATH:STRING=${llvm_lib_dir}" \
    "${extra_opts[@]}"
fi

status "Building Torch-MLIR (Ninja)"
cmake --build "$torch_mlir_build_dir" --target all TorchMLIRPythonModules

verbose_cmd cmake --install "$torch_mlir_build_dir" --prefix "$torch_mlir_install_dir"

status "Building and installing Torch-MLIR Python package for $python_for_install"

# With LLVM_INSTALL_DIR set, setup.py reads the package straight out of
# <build>/python_packages/torch_mlir, which is where an out-of-tree build puts
# it. Without it, setup.py would instead look under
# <build>/tools/torch-mlir/python_packages, the in-tree layout.
python_package_dir="$torch_mlir_build_dir/python_packages/torch_mlir"

# MLIR's type stubs used to be checked into the LLVM sources and were symlinked
# into this tree; since llvm efd96afedf they are generated into LLVM's build dir
# instead, which leaves the old symlinks dangling. CMake will not replace a
# symlink that already exists, so a reconfigure does not clear them, and
# setup.py fails trying to copy them. They are only type hints, so drop any that
# no longer resolve.
if [[ -d "$python_package_dir" ]]; then
  while IFS= read -r stale; do
    info "Removing stale symlink $stale"
    rm -f "$stale"
  done < <(find "$python_package_dir" -xtype l)
fi

pushd "$torch_mlir_source_dir" >/dev/null
TORCH_MLIR_CMAKE_ALREADY_BUILT=1 TORCH_MLIR_CMAKE_BUILD_DIR="$torch_mlir_build_dir" LLVM_INSTALL_DIR="$llvm_build_dir" PYTHONWARNINGS=ignore \
  verbose_cmd "$python_for_install" -m pip install --no-build-isolation --no-deps --force-reinstall .
popd >/dev/null
