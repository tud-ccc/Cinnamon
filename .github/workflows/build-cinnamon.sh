#!/usr/bin/env bash
set -euo pipefail

script_dir="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
# shellcheck source=/dev/null
source "$script_dir/common.sh"

# ---- Safe defaults to avoid 'unbound variable' ----
reconfigure="${reconfigure:-0}"
setup_python_venv="${setup_python_venv:-0}"
checkout_and_build_llvm="${checkout_and_build_llvm:-0}"
checkout_and_build_torch_mlir="${checkout_and_build_torch_mlir:-0}"
checkout_upmem="${checkout_upmem:-0}"
CINNAMON_CMAKE_OPTIONS="${CINNAMON_CMAKE_OPTIONS:-}"
CINNAMON_BUILD_OPTIONS="${CINNAMON_BUILD_OPTIONS:-}"

# Required paths (defined in common.sh)
project_root="${project_root:?Define 'project_root' in common.sh}"
cinnamon_path="${cinnamon_path:?Define 'cinnamon_path' in common.sh}"
llvm_path="${llvm_path:-}"
torch_mlir_path="${torch_mlir_path:-}"
upmem_path="${upmem_path:-}"

# ---- Tools ----
command -v ninja >/dev/null 2>&1 || { error "Ninja not found. Install it (e.g., 'sudo apt install ninja-build')."; exit 1; }
command -v cmake >/dev/null 2>&1 || { error "CMake not found. Install it."; exit 1; }

cd "$cinnamon_path"


# ---- Build dir sanity: reconfigure if wrong generator / missing files ----
need_config=0
reason=""

if [[ -f build/CMakeCache.txt ]] && ! grep -q 'CMAKE_GENERATOR:INTERNAL=Ninja' build/CMakeCache.txt; then
  reason="existing build is not Ninja"
fi
if [[ -z "$reason" && ! -d build ]]; then reason="build/ directory missing"; fi
if [[ -z "$reason" && -d build && ! -f build/CMakeCache.txt ]]; then reason="CMakeCache.txt missing"; fi
if [[ -z "$reason" && -d build && ! -f build/build.ninja ]]; then reason="build.ninja missing"; fi
if [[ -z "$reason" && "$reconfigure" -eq 1 ]]; then reason="forced reconfigure (reconfigure=1)"; fi

# ---- If a venv is active, make CMake use it (Python + pybind11) ----
EXTRA_OPTS=()
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PYBIN="$(command -v python)"
  # Ensure pybind11 is available and get its cmake dir
  if ! PYBIND11_DIR="$("$PYBIN" - <<'PY'
import sys
try:
    import pybind11
    print(pybind11.get_cmake_dir())
except Exception:
    sys.exit(1)
PY
)"; then
    status "pybind11 not found in venv; installing…"
    "$PYBIN" -m pip install -U "pybind11>=2.10" numpy >/dev/null
    PYBIND11_DIR="$("$PYBIN" -c 'import pybind11; print(pybind11.get_cmake_dir())')"
  fi
  EXTRA_OPTS+=( -DPython3_EXECUTABLE="$PYBIN" -Dpybind11_DIR="$PYBIND11_DIR" -DPython3_FIND_VIRTUALENV=ONLY )

  # If cache exists but uses a different Python, force reconfigure
  if [[ -z "$reason" && -f build/CMakeCache.txt ]]; then
    cached_py="$(grep -E '^Python3_EXECUTABLE:FILEPATH=' build/CMakeCache.txt | sed 's/.*=//')"
    if [[ -n "${cached_py:-}" && "$cached_py" != "$PYBIN" ]]; then
      reason="cached Python ($cached_py) != venv Python ($PYBIN)"
    fi
  fi
fi

# ---- Dependency locations (assembled safely as an array) ----
DEP_OPTS=()
if [[ "$checkout_and_build_llvm" -eq 1 && -n "${llvm_path:-}" ]]; then
  DEP_OPTS+=( -DLLVM_DIR="$llvm_path/build/lib/cmake/llvm" )
  DEP_OPTS+=( -DMLIR_DIR="$llvm_path/build/lib/cmake/mlir" )
fi
if [[ "$checkout_upmem" -eq 1 && -n "${upmem_path:-}" ]]; then
  DEP_OPTS+=( -DUPMEM_DIR="$upmem_path" )
fi
if [[ "$checkout_and_build_torch_mlir" -eq 1 && -n "${torch_mlir_path:-}" ]]; then
  DEP_OPTS+=( -DTORCH_MLIR_DIR="$torch_mlir_path/install" )
fi

# User-provided extra options (space-separated → array)
EXTRA_USER_OPTS=()
if [[ -n "$CINNAMON_CMAKE_OPTIONS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_USER_OPTS=( $CINNAMON_CMAKE_OPTIONS )
fi

# ---- Configure helper ----
configure() {
  status "Configuring Cinnamon (Ninja)"
  ln -s "$project_root/LICENSE" "$cinnamon_path/python/" 2>/dev/null || true

  BUILD_TYPE=${CMAKE_BUILD_TYPE:=RelWithDebInfo}

  # ---- Conan: install C++ dependencies into build/ ----
  if command -v conan >/dev/null 2>&1 && [[ -f conanfile.txt ]]; then
    status "Running conan install"
    mkdir -p build
    conan install . --output-folder=build --build=missing -s build_type=${BUILD_TYPE} -s compiler.cppstd=20
  else
    warning "conan not found or no conanfile.txt — skipping conan install"
  fi

  # pushd build/$BUILD_TYPE
  local cmake_args=(
    -S .
    -B build
    -G Ninja
    -DCMAKE_TOOLCHAIN_FILE=build/conan_toolchain.cmake
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE}
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    -DLLVM_ENABLE_EH=ON
    -DLLVM_ENABLE_RTTI=ON
  )

  if ((${#DEP_OPTS[@]})); then
    cmake_args+=("${DEP_OPTS[@]}")
  fi
  if ((${#EXTRA_OPTS[@]})); then
    cmake_args+=("${EXTRA_OPTS[@]}")
  fi
  if ((${#EXTRA_USER_OPTS[@]})); then
    cmake_args+=("${EXTRA_USER_OPTS[@]}")
  fi

  source build/conanbuild.sh
  print_and_run cmake "${cmake_args[@]}"
  cmake --build build --target all $CINNAMON_BUILD_OPTIONS
  source deactivate_conanbuild.sh
  popd
}

# ---- Build with one clean retry on failure ----
status "Building Cinnamon (Ninja)"
if ! cmake --build build --target all $CINNAMON_BUILD_OPTIONS; then
  warning "Build failed — cleaning build/ and retrying from fresh configure…"
  rm -rf build
  configure
fi

# ---- Python package wiring (optional) ----
if [[ "$setup_python_venv" -eq 1 && -n "${llvm_path:-}" && -n "${torch_mlir_path:-}" ]]; then
  status "Building Cinnamon Python package"
  # Prefer sysconfig (distutils may be absent)
  site_packages_dir="$(python - <<'PY'
import sys, sysconfig
print(sysconfig.get_paths().get("platlib") or sysconfig.get_paths().get("purelib"))
PY
)"
  cinnamon_python_package_dir_src="$project_root/python/src/cinnamon"
  cinnamon_python_package_dir_dest="$site_packages_dir"
  cinnamon_python_package_resource_dir="$cinnamon_python_package_dir_dest/_resources"

  cinnamon_python_resources=()
  cinnamon_python_resources+=( "$cinnamon_path/build/bin/cinm-opt" )
  cinnamon_python_resources+=( "$cinnamon_path/build/lib/libMemristorDialectRuntime.so" )
  [[ -n "${torch_mlir_path:-}" ]] && cinnamon_python_resources+=( "$torch_mlir_path/build/bin/torch-mlir-opt" )
  [[ -n "${llvm_path:-}" ]] && cinnamon_python_resources+=( "$llvm_path/build/bin/mlir-translate" )
  [[ -n "${llvm_path:-}" ]] && cinnamon_python_resources+=( "$llvm_path/build/bin/clang" )

  if [[ ! -e "$cinnamon_python_package_dir_dest" ]]; then
    ln -s "$cinnamon_python_package_dir_src" "$cinnamon_python_package_dir_dest"
  fi

  mkdir -p "$cinnamon_python_package_resource_dir" || true
  for resource in "${cinnamon_python_resources[@]}"; do
    ln -s "$resource" "$cinnamon_python_package_resource_dir" 2>/dev/null || true
  done

  if [[ "${build_cinnamon_wheel:-0}" -eq 1 ]]; then
    cd "$cinnamon_path/python"
    PYTHONWARNINGS=ignore verbose_cmd python -m build
  fi
else
  warning "Skipping Cinnamon Python package build"
  warning "Ensure your Python env is set up if you need it."
fi
