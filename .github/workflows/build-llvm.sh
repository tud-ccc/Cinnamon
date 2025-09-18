#!/usr/bin/env bash
set -euo pipefail

script_dir="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
source "$script_dir/common.sh"

# ---- Safe defaults ----
checkout_and_build_llvm="${checkout_and_build_llvm:-0}"
reconfigure="${reconfigure:-0}"
llvm_path="${llvm_path:?Define 'llvm_path' in common.sh}"
LLVM_CMAKE_OPTIONS="${LLVM_CMAKE_OPTIONS:-}"

# Tools
command -v ninja >/dev/null 2>&1 || { error "Ninja not found."; exit 1; }
command -v cmake >/dev/null 2>&1 || { error "CMake not found."; exit 1; }

# Collect extra cmake opts (space-separated env -> array)
EXTRA_CMAKE_OPTS=()
if [[ -n "$LLVM_CMAKE_OPTIONS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_CMAKE_OPTS=( $LLVM_CMAKE_OPTIONS )
fi

# If in a venv, ensure CMake uses that Python and finds pybind11
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PYBIN="$(command -v python)"
  # Ensure pybind11 is present and get its CMake dir
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
    python -m pip install -U "pybind11>=2.10" numpy >/dev/null
    PYBIND11_DIR="$("$PYBIN" -c 'import pybind11; print(pybind11.get_cmake_dir())')"
  fi
  EXTRA_CMAKE_OPTS+=( -DPython3_EXECUTABLE="$PYBIN" -Dpybind11_DIR="$PYBIND11_DIR" -DPython3_FIND_VIRTUALENV=ONLY )
fi

if [[ "$checkout_and_build_llvm" -ne 1 ]]; then
  warning "Skipping LLVM checkout and build (set checkout_and_build_llvm=1)."
  exit 0
fi

# ---- Clone if missing ----
need_config=0
if [[ ! -d "$llvm_path" ]]; then
  status "Checking out LLVM"
  git clone https://github.com/h4midf/llvm-project.git --depth 1 --branch cinnamon-esweek-llvm "$llvm_path"
  need_config=1
else
  status "Found existing LLVM at: $llvm_path"
fi

pushd "$llvm_path" >/dev/null

# ---- Decide whether to (re)configure ----
reason=""
if [[ "$reconfigure" -eq 1 ]]; then
  reason="forced reconfigure (reconfigure=1)"
fi

# Wrong generator -> wipe
if [[ -z "$reason" && -f build/CMakeCache.txt && ! "$(grep -o 'CMAKE_GENERATOR:INTERNAL=[^ ]*' build/CMakeCache.txt || true)" =~ Ninja ]]; then
  reason="existing build is not Ninja"
fi

# Missing build dir / cache / build.ninja
if [[ -z "$reason" && ! -d build ]]; then reason="build/ directory missing"; fi
if [[ -z "$reason" && ! -f build/CMakeCache.txt ]]; then reason="CMakeCache.txt missing"; fi
if [[ -z "$reason" && ! -f build/build.ninja ]]; then reason="build.ninja missing"; fi

# Cached Python mismatch with venv Python
if [[ -z "$reason" && -f build/CMakeCache.txt && -n "${PYBIN:-}" ]]; then
  cached_py="$(grep -E '^Python3_EXECUTABLE:FILEPATH=' build/CMakeCache.txt | sed 's/.*=//')"
  if [[ -n "$cached_py" && "$cached_py" != "$PYBIN" ]]; then
    reason="cached Python ($cached_py) != venv Python ($PYBIN)"
  fi
fi

if [[ -n "$reason" ]]; then
  status "Reconfiguring because: $reason"
  rm -rf build
  status "Configuring LLVM (Ninja)"
  cmake -S llvm -B build -G Ninja \
    -Wno-dev \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DLLVM_BUILD_TOOLS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DLLVM_OPTIMIZED_TABLEGEN=ON \
    -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=SPIRV \
    "${EXTRA_CMAKE_OPTS[@]}"
else
  status "Using existing LLVM configuration in build/"
fi

# ---- Build with one automatic clean-retry ----
status "Building LLVM (Ninja)"
if ! cmake --build build --target all llc opt mlir-opt mlir-translate; then
  warning "Build failed — cleaning build/ and retrying from fresh configure…"
  rm -rf build
  status "Reconfiguring after failure"
  cmake -S llvm -B build -G Ninja \
    -Wno-dev \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DLLVM_BUILD_TOOLS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DLLVM_OPTIMIZED_TABLEGEN=ON \
    -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=SPIRV \
    "${EXTRA_CMAKE_OPTS[@]}"
  cmake --build build --target all llc opt mlir-opt mlir-translate
fi

export PATH="$llvm_path/build/bin:$PATH"
popd >/dev/null
