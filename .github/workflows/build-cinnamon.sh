#!/usr/bin/env bash
set -euo pipefail

script_dir="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
# shellcheck source=/dev/null
source "$script_dir/common.sh"

CINNAMON_CMAKE_OPTIONS="${CINNAMON_CMAKE_OPTIONS:-}"
CINNAMON_BUILD_OPTIONS="${CINNAMON_BUILD_OPTIONS:-}"

command -v ninja >/dev/null 2>&1 || { error "Ninja not found. Install it (e.g., 'sudo apt install ninja-build')."; exit 1; }
command -v cmake >/dev/null 2>&1 || { error "CMake not found. Install it."; exit 1; }

# The C++ cost model predictor is built as part of this project.
ensure_submodule third-party/cnm-cost-model

cd "$cinnamon_path"

cache_file="$cinnamon_build_dir/CMakeCache.txt"

# ---- Where our dependencies live ----
dep_opts=(
  -DLLVM_DIR="$llvm_cmake_dir"
  -DMLIR_DIR="$mlir_cmake_dir"
)
if [[ "$enable_upmem" -eq 1 && -d "$upmem_dir" ]]; then
  dep_opts+=( -DUPMEM_DIR="$upmem_dir" )
fi
if [[ -d "$torch_mlir_install_dir" ]]; then
  dep_opts+=( -DTORCH_MLIR_DIR="$torch_mlir_install_dir" )
else
  warning "No Torch-MLIR installation at '$torch_mlir_install_dir'; the torch frontend will not be built"
fi

# ---- If a venv is active, make CMake use it (Python + pybind11) ----
python_opts=()
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  if [[ -z "${PYBIND11_DIR:-}" ]]; then
    status "pybind11 not found in venv; installing..."
    verbose_cmd "$PYBIN" -m pip install -U "pybind11>=2.10" numpy
    PYBIND11_DIR="$("$PYBIN" -c 'import pybind11; print(pybind11.get_cmake_dir())')"
  fi
  python_opts=( -DPython3_EXECUTABLE="$PYBIN" -Dpybind11_DIR="$PYBIND11_DIR" -DPython3_FIND_VIRTUALENV=ONLY )
fi

# ---- Decide whether we need to configure ----
need_config=0
reason=""
if [[ ! -f "$cache_file" ]]; then
  reason="no CMake cache in '$cinnamon_build_dir'"
elif ! grep -q 'CMAKE_GENERATOR:INTERNAL=Ninja' "$cache_file"; then
  reason="existing build is not Ninja"
elif [[ ! -f "$cinnamon_build_dir/build.ninja" ]]; then
  reason="build.ninja missing"
elif [[ "$reconfigure" -eq 1 ]]; then
  reason="forced reconfigure (reconfigure=1)"
elif [[ -n "${PYBIN:-}" ]]; then
  cached_py="$(grep -E '^Python3_EXECUTABLE:FILEPATH=' "$cache_file" | sed 's/.*=//' || true)"
  if [[ -n "$cached_py" && "$cached_py" != "$PYBIN" ]]; then
    reason="cached Python ($cached_py) != venv Python ($PYBIN)"
  fi
fi
[[ -n "$reason" ]] && need_config=1

BUILD_TYPE="${CMAKE_BUILD_TYPE:-RelWithDebInfo}"

if [[ "$need_config" -eq 1 ]]; then
  status "Configuring Cinnamon (Ninja): $reason"
  ln -s "$project_root/LICENSE" "$cinnamon_path/python/" 2>/dev/null || true

  # ---- Conan: install C++ dependencies into the build dir ----
  if ! command -v conan >/dev/null 2>&1; then
    error "conan not found. Run setup-venv.sh, or install conan into the active environment."
    exit 1
  fi
  status "Running conan install"
  mkdir -p "$cinnamon_build_dir"
  verbose_cmd conan install . --output-folder="$cinnamon_build_dir" --build=missing \
    -s build_type="$BUILD_TYPE" -pr "$project_root/third-party/conan-profile"

  user_opts=()
  if [[ -n "$CINNAMON_CMAKE_OPTIONS" ]]; then
    # shellcheck disable=SC2206
    user_opts=( $CINNAMON_CMAKE_OPTIONS )
  fi

  # shellcheck disable=SC1091
  source "$cinnamon_build_dir/conanbuild.sh"
  print_and_run cmake -S "$cinnamon_path" -B "$cinnamon_build_dir" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$cinnamon_build_dir/conan_toolchain.cmake" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DLLVM_ENABLE_EH=ON \
    -DLLVM_ENABLE_RTTI=ON \
    "${dep_opts[@]}" \
    ${python_opts[@]+"${python_opts[@]}"} \
    ${user_opts[@]+"${user_opts[@]}"}
  # shellcheck disable=SC1091
  source "$cinnamon_build_dir/deactivate_conanbuild.sh"
fi

status "Building Cinnamon (Ninja)"
# shellcheck disable=SC2086
print_and_run cmake --build "$cinnamon_build_dir" --target all $CINNAMON_BUILD_OPTIONS

# ---- Python package wiring ----
if [[ "$setup_python_venv" -eq 1 ]]; then
  status "Building Cinnamon Python package"
  site_packages_dir="$(python -c 'import sysconfig; p=sysconfig.get_paths(); print(p.get("platlib") or p.get("purelib"))')"
  cinnamon_python_package_dir_src="$project_root/python/src/cinnamon"
  cinnamon_python_package_resource_dir="$site_packages_dir/_resources"

  cinnamon_python_resources=(
    "$cinnamon_build_dir/bin/cinm-opt"
    "$cinnamon_build_dir/lib/libMemristorDialectRuntime.so"
    "$torch_mlir_build_dir/bin/torch-mlir-opt"
    "$llvm_build_dir/bin/mlir-translate"
    "$llvm_build_dir/bin/clang"
  )

  if [[ ! -e "$site_packages_dir" ]]; then
    ln -s "$cinnamon_python_package_dir_src" "$site_packages_dir"
  fi

  mkdir -p "$cinnamon_python_package_resource_dir" || true
  for resource in "${cinnamon_python_resources[@]}"; do
    ln -s "$resource" "$cinnamon_python_package_resource_dir" 2>/dev/null || true
  done

  if [[ "$build_cinnamon_wheel" -eq 1 ]]; then
    pushd "$cinnamon_path/python" >/dev/null
    PYTHONWARNINGS=ignore verbose_cmd python -m build
    popd >/dev/null
  fi
else
  warning "Skipping Cinnamon Python package build"
  warning "Ensure your Python env is set up if you need it."
fi
