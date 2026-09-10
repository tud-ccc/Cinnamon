#!/usr/bin/env bash
set -euo pipefail

script_dir="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
# shellcheck source=/dev/null
source "$script_dir/common.sh"

LLVM_CMAKE_OPTIONS="${LLVM_CMAKE_OPTIONS:-}"

# Desired config (override via env if needed)
LLVM_PROJECTS="${LLVM_PROJECTS:-mlir;llvm}"
LLVM_TARGETS_TO_BUILD="${LLVM_TARGETS_TO_BUILD:-host;AArch64}"
LLVM_EXPERIMENTAL_TARGETS="${LLVM_EXPERIMENTAL_TARGETS:-SPIRV}"
LLVM_BUILD_TARGETS="${LLVM_BUILD_TARGETS:-all llc opt mlir-opt mlir-translate}"

if [[ "$build_llvm" -eq 0 ]]; then
  status "Not building LLVM; using '$llvm_build_dir'"
  export PATH="$llvm_build_dir/bin:$PATH"
else

# The submodule is only needed when we build LLVM ourselves from the in-tree
# sources. LLVM's history is large, so fetch it shallowly.
if [[ -z "${LLVM_SOURCE_DIR:-}" ]]; then
  ensure_submodule third-party/llvm 1
fi

command -v ninja >/dev/null 2>&1 || { error "Ninja not found."; exit 1; }
command -v cmake >/dev/null 2>&1 || { error "CMake not found."; exit 1; }

# Extra cmake opts (space-separated env -> array)
EXTRA_CMAKE_OPTS=()
if [[ -n "$LLVM_CMAKE_OPTIONS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_CMAKE_OPTS=( $LLVM_CMAKE_OPTIONS )
fi

cache_file="$llvm_build_dir/CMakeCache.txt"

# ---- Should we clean the build dir? ----
clean_reason=""
if [[ "$reconfigure" -eq 1 ]]; then
  clean_reason="forced reconfigure (reconfigure=1)"
elif [[ -d "$llvm_build_dir" && ! -f "$cache_file" ]] && [[ -n "$(ls -A "$llvm_build_dir" 2>/dev/null)" ]]; then
  # Output present but no cache: a previous configure was interrupted or failed.
  clean_reason="previous configure left the build dir incomplete"
elif [[ -f "$cache_file" && ! "$(grep -o 'CMAKE_GENERATOR:INTERNAL=[^ ]*' "$cache_file" || true)" =~ Ninja ]]; then
  clean_reason="existing build is not Ninja"
elif [[ -f "$cache_file" && -n "${PYBIN:-}" ]]; then
  cached_py="$(grep -E '^Python3_EXECUTABLE:(FILEPATH|UNINITIALIZED)=' "$cache_file" | sed 's/.*=//' || true)"
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

mkdir -p "$llvm_build_dir"
HASH_FILE="$llvm_build_dir/.config.hash"
if [[ -z "$clean_reason" && -f "$HASH_FILE" ]]; then
  OLD_HASH="$(cat "$HASH_FILE" 2>/dev/null || true)"
  [[ "$OLD_HASH" != "$CURRENT_HASH" ]] && clean_reason="configuration changed (hash mismatch)"
fi

if [[ -n "$clean_reason" ]]; then
  status "Cleaning '$llvm_build_dir' because: $clean_reason"
  rm -rf "$llvm_build_dir"
  mkdir -p "$llvm_build_dir"
fi

status "Configuring LLVM (Ninja; always run to catch changes)"
print_and_run cmake -S "$llvm_source_dir/llvm" -B "$llvm_build_dir" -G Ninja \
  -Wno-dev \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DLLVM_BUILD_TOOLS=ON \
  -DLLVM_CCACHE_BUILD=ON \
  -DLLVM_ENABLE_PROJECTS="$LLVM_PROJECTS" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_EH=ON \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD="$LLVM_EXPERIMENTAL_TARGETS" \
  -DLLVM_INCLUDE_BENCHMARKS=OFF \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DLLVM_OPTIMIZED_TABLEGEN=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_TARGETS_TO_BUILD="$LLVM_TARGETS_TO_BUILD" \
  "${EXTRA_CMAKE_OPTS[@]}"

# Save config hash so we can detect future changes
echo "$CURRENT_HASH" > "$HASH_FILE"

[[ -f "$llvm_build_dir/build.ninja" ]] || { error "CMake configure did not produce build.ninja."; exit 1; }

status "Building LLVM (Ninja)"
# shellcheck disable=SC2086
cmake --build "$llvm_build_dir" --target ${LLVM_BUILD_TARGETS}

export PATH="$llvm_build_dir/bin:$PATH"

fi
