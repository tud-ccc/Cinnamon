#!/usr/bin/env bash
set -euo pipefail

script_dir="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
# shellcheck source=/dev/null
source "$script_dir/common.sh"

# ---- Safe defaults ----
checkout_and_build_llvm="${checkout_and_build_llvm:-1}"
reconfigure="${reconfigure:-0}"
llvm_path="${llvm_path:?Define 'llvm_path' in common.sh}"
LLVM_CMAKE_OPTIONS="${LLVM_CMAKE_OPTIONS:-}"

# Your desired config (override via env if needed)
LLVM_PROJECTS="${LLVM_PROJECTS:-mlir;llvm}"
LLVM_TARGETS_TO_BUILD="${LLVM_TARGETS_TO_BUILD:-host;AArch64}"
LLVM_EXPERIMENTAL_TARGETS="${LLVM_EXPERIMENTAL_TARGETS:-SPIRV}"
LLVM_BUILD_TARGETS="${LLVM_BUILD_TARGETS:-all llc opt mlir-opt mlir-translate}"

# Tools
command -v ninja >/dev/null 2>&1 || { error "Ninja not found."; exit 1; }
command -v cmake >/dev/null 2>&1 || { error "CMake not found."; exit 1; }

# Extra cmake opts (space-separated env -> array)
EXTRA_CMAKE_OPTS=()
if [[ -n "$LLVM_CMAKE_OPTIONS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_CMAKE_OPTS=( $LLVM_CMAKE_OPTIONS )
fi

# If in a venv, force using that Python + pybind11
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PYBIN="$(command -v python)"
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
  EXTRA_CMAKE_OPTS+=( -DPython3_EXECUTABLE="$PYBIN" -Dpybind11_DIR="$PYBIND11_DIR" -DPython3_FIND_VIRTUALENV=ONLY )
fi

# ---- Clone if missing (only when requested) ----
if [[ ! -d "$llvm_path" ]]; then
  if [[ "$checkout_and_build_llvm" -eq 1 ]]; then
    status "Checking out LLVM"
    git clone https://github.com/h4midf/llvm-project.git --depth 1 --branch cinnamon-esweek-llvm "$llvm_path"
  else
    error "LLVM path '$llvm_path' does not exist. Set checkout_and_build_llvm=1 to clone, or create it manually."
    exit 1
  fi
else
  status "Found existing LLVM at: $llvm_path"
fi

if [[ "${checkout_and_build_llvm}" -eq 0 ]]; then
  status "Not rebuilding/reconfiguring LLVM."
  export PATH="$llvm_path/build/bin:$PATH"
  return 0
fi

pushd "$llvm_path" >/dev/null


# ---- Should we clean build/? ----
clean_reason=""
if [[ "${reconfigure}" -eq 1 && ! "${checkout_and_build_llvm}" -eq 0 ]]; then
  clean_reason="forced reconfigure (reconfigure=1)"
elif [[ -f build/CMakeCache.txt && ! "$(grep -o 'CMAKE_GENERATOR:INTERNAL=[^ ]*' build/CMakeCache.txt || true)" =~ Ninja ]]; then
  clean_reason="existing build is not Ninja"
elif [[ -f build/CMakeCache.txt && -n "${PYBIN:-}" ]]; then
  cached_py="$(grep -E '^Python3_EXECUTABLE:(FILEPATH|UNINITIALIZED)=' build/CMakeCache.txt | sed 's/.*=//' || true)"
  [[ -n "$cached_py" && "$cached_py" != "${PYBIN:-}" ]] && clean_reason="cached Python ($cached_py) != venv Python (${PYBIN:-system})"
fi

# Also clean if config hash changed (projects/targets/opts/python/etc.)
hash_cmd() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum; elif command -v shasum >/dev/null 2>&1; then shasum -a 256; else python - <<'PY'
import sys,hashlib
data=sys.stdin.read().encode();print(hashlib.sha256(data).hexdigest())
PY
  fi
}
EXTRA_CMAKE_OPTS_JOIN="${EXTRA_CMAKE_OPTS[*]}"
CURRENT_HASH="$(printf '%s\n' \
  "PROJ=$LLVM_PROJECTS" \
  "TGT=$LLVM_TARGETS_TO_BUILD" \
  "EXP=$LLVM_EXPERIMENTAL_TARGETS" \
  "OPTS=${EXTRA_CMAKE_OPTS_JOIN}" \
  "PY=${PYBIN:-}" \
  "GEN=Ninja" | hash_cmd | awk '{print $1}')"

mkdir -p build
HASH_FILE="build/.config.hash"
if [[ -z "$clean_reason" && -f "$HASH_FILE" ]]; then
  OLD_HASH="$(cat "$HASH_FILE" 2>/dev/null || true)"
  [[ "$OLD_HASH" != "$CURRENT_HASH" ]] && clean_reason="configuration changed (hash mismatch)"
fi

if [[ -n "$clean_reason" ]]; then
  status "Cleaning build/ because: $clean_reason"
  rm -rf build
  mkdir -p build
fi
# ---- Always run configure (idempotent) ----
status "Configuring LLVM (Ninja; always run to catch changes)"
print_and_run cmake -S llvm -B build -G Ninja \
  -Wno-dev \
  -DLLVM_ENABLE_PROJECTS="$LLVM_PROJECTS" \
  -DLLVM_TARGETS_TO_BUILD="$LLVM_TARGETS_TO_BUILD" \
  -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD="$LLVM_EXPERIMENTAL_TARGETS" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_BUILD_TOOLS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DLLVM_INCLUDE_BENCHMARKS=OFF \
  -DLLVM_OPTIMIZED_TABLEGEN=ON \
  -DLLVM_CCACHE_BUILD=ON \
  -DLLVM_PARALLEL_COMPILE_JOBS=4 \
  -DLLVM_PARALLEL_LINK_JOBS=1 \
  -DLLVM_PARALLEL_TABLEGEN_JOBS=4 \
  "${EXTRA_CMAKE_OPTS[@]}"

# Save config hash so we can detect future changes
echo "$CURRENT_HASH" > "$HASH_FILE"

# Sanity: ensure build.ninja exists
[[ -f build/build.ninja ]] || { error "CMake configure did not produce build/build.ninja."; exit 1; }

# ---- Build with one clean-retry ----
status "Building LLVM (Ninja)"
if ! cmake --build build --target ${LLVM_BUILD_TARGETS}; then
  warning "Build failed — cleaning build/ and retrying from fresh configure…"
  rm -rf build
  cmake -S llvm -B build -G Ninja \
    -Wno-dev \
    -DLLVM_ENABLE_PROJECTS="$LLVM_PROJECTS" \
    -DLLVM_TARGETS_TO_BUILD="$LLVM_TARGETS_TO_BUILD" \
    -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD="$LLVM_EXPERIMENTAL_TARGETS" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DLLVM_BUILD_TOOLS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DLLVM_OPTIMIZED_TABLEGEN=ON \
    -DLLVM_CCACHE_BUILD=ON \
    -DLLVM_PARALLEL_COMPILE_JOBS=4 \
    -DLLVM_PARALLEL_LINK_JOBS=1 \
    -DLLVM_PARALLEL_TABLEGEN_JOBS=4 \
    "${EXTRA_CMAKE_OPTS[@]}"
  cmake --build build --target ${LLVM_BUILD_TARGETS}
fi

export PATH="$llvm_path/build/bin:$PATH"
popd >/dev/null
