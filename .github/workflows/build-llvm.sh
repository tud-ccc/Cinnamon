#!/usr/bin/env bash
set -euo pipefail

script_dir="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
# shellcheck source=/dev/null
source "$script_dir/common.sh"

LLVM_CMAKE_OPTIONS="${LLVM_CMAKE_OPTIONS:-}"

# Overrides of the configuration from the environment. By default, it comes
# from the LLVM sources: see below.
llvm_overrides=()
[[ -z "${LLVM_PROJECTS:-}" ]] || llvm_overrides+=( -DLLVM_ENABLE_PROJECTS="$LLVM_PROJECTS" )
[[ -z "${LLVM_TARGETS_TO_BUILD:-}" ]] || llvm_overrides+=( -DLLVM_TARGETS_TO_BUILD="$LLVM_TARGETS_TO_BUILD" )
[[ -z "${LLVM_EXPERIMENTAL_TARGETS:-}" ]] || llvm_overrides+=( -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD="$LLVM_EXPERIMENTAL_TARGETS" )

LLVM_PROJECTS="${LLVM_PROJECTS:-mlir;llvm}"
LLVM_TARGETS_TO_BUILD="${LLVM_TARGETS_TO_BUILD:-host;AArch64}"
LLVM_EXPERIMENTAL_TARGETS="${LLVM_EXPERIMENTAL_TARGETS:-SPIRV}"
LLVM_BUILD_TARGETS="${LLVM_BUILD_TARGETS:-all llc opt mlir-opt mlir-translate}"

# Download the prebuilt LLVM for the pinned revision, and unpack it into
# $llvm_prebuilt_dir. Fails, leaving that directory as it was, if there is none.
#
# This runs as an `if` condition, where `set -e` does not apply, so every step
# checks for failure itself.
download_dir=""
trap '[[ -z "$download_dir" ]] || rm -rf "$download_dir"' EXIT
fetch_prebuilt_llvm() {
  if [[ -z "$llvm_revision" ]]; then
    warning "Cannot tell which LLVM revision third-party/llvm is pinned to"
    return 1
  fi
  if [[ "$(cat "$llvm_prebuilt_dir/$llvm_prebuilt_stamp" 2>/dev/null)" == "$llvm_revision" ]]; then
    info "Prebuilt LLVM ${llvm_revision:0:12} is already in '$llvm_prebuilt_dir'"
    return 0
  fi
  if [[ "$(uname -s)/$(uname -m)" != Linux/x86_64 ]]; then
    warning "Prebuilt LLVM is only published for Linux on x86_64"
    return 1
  fi
  local tool
  for tool in curl zstd sha256sum; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      warning "Downloading a prebuilt LLVM needs '$tool', which was not found"
      return 1
    fi
  done

  local name url archive
  name="$(llvm_prebuilt_name "$llvm_revision")"
  url="$(llvm_prebuilt_url "$llvm_revision")"
  # Next to its destination, so that moving it into place is a rename
  download_dir="$(mktemp -d "$project_root/third-party/.llvm-prebuilt.XXXXXX")" || return 1
  archive="$download_dir/$name.tar.zst"

  status "Downloading prebuilt LLVM ${llvm_revision:0:12} from $url"
  if ! curl -fL --retry 3 --progress-bar -o "$archive" "$url"; then
    warning "There is no prebuilt LLVM for ${llvm_revision:0:12}"
    return 1
  fi
  if ! curl -fsSL --retry 3 -o "$archive.sha256" "$url.sha256" \
      || ! (cd "$download_dir" && sha256sum --quiet -c "$name.tar.zst.sha256"); then
    error "The checksum of '$url' does not match"
    return 1
  fi
  status "Unpacking it into '$llvm_prebuilt_dir'"
  if ! zstd -dc "$archive" | tar -x -C "$download_dir"; then
    error "Could not unpack '$archive'"
    return 1
  fi
  if [[ "$(cat "$download_dir/$name/$llvm_prebuilt_stamp" 2>/dev/null)" != "$llvm_revision" ]]; then
    error "'$url' does not contain LLVM $llvm_revision"
    return 1
  fi
  rm -rf "$llvm_prebuilt_dir" && mv "$download_dir/$name" "$llvm_prebuilt_dir"
}

if [[ "$build_llvm" -eq 0 ]]; then
  status "Not building LLVM; using '$llvm_build_dir'"
  export PATH="$llvm_build_dir/bin:$PATH"
else

if [[ "$use_prebuilt_llvm" -eq 1 ]]; then
  if fetch_prebuilt_llvm; then
    # Its MLIR Python bindings only load into the Python version they were
    # built for, and Torch-MLIR builds on them.
    prebuilt_python="$(cat "$llvm_prebuilt_dir/cinnamon-llvm-python" 2>/dev/null || true)"
    our_python="$("${PYBIN:-python3}" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
    if [[ -n "$prebuilt_python" && "$prebuilt_python" != "$our_python" ]]; then
      warning "The prebuilt LLVM has Python bindings for Python $prebuilt_python, but ours is ${our_python:-missing}."
      warning "Torch-MLIR will not build against them; use Python $prebuilt_python, or LLVM_PREBUILT=never."
    fi
    # MLIR identifies traits and interfaces by addresses that compilers do not
    # share, so a prebuilt LLVM only works with the compiler family it was
    # built with. Mixing them gives passes that cannot see attributes and
    # interfaces which are plainly there, and tests that fail far from the
    # cause. A clang-built library names GCC too, for its startup files, so
    # clang wins when both appear.
    prebuilt_lib="$(find "$llvm_prebuilt_dir/lib" -maxdepth 1 -name 'libMLIRIR.so*' -print -quit 2>/dev/null || true)"
    if [[ -n "$prebuilt_lib" ]] && command -v readelf >/dev/null 2>&1; then
      comment="$(readelf -p .comment "$prebuilt_lib" 2>/dev/null || true)"
      if [[ "$comment" == *"clang version"* ]]; then
        their_cc="clang $(grep -m1 -oE 'clang version [0-9]+' <<<"$comment" | cut -d' ' -f3)"
      elif [[ "$comment" == *"GCC:"* ]]; then
        their_cc="gcc $(grep -m1 -oE 'GCC: \([^)]*\) [0-9]+' <<<"$comment" | sed 's/.* //')"
      else
        their_cc=""
      fi
      cxx_version_line="$("$CXX" --version 2>/dev/null | sed -n 1p)"
      case "$cxx_version_line" in
        *[Cc]lang*) our_cc="clang $("$CXX" -dumpversion 2>/dev/null | cut -d. -f1)" ;;
        *)          our_cc="gcc $("$CXX" -dumpversion 2>/dev/null | cut -d. -f1)" ;;
      esac
      if [[ -n "$their_cc" && "$their_cc" != "$our_cc" ]]; then
        error "The prebuilt LLVM was built with $their_cc, but this build uses $our_cc ($CXX)."
        error "MLIR identifies traits and interfaces by addresses that compilers do not share,"
        error "so the two cannot be mixed: passes stop seeing attributes that are plainly there."
        error "Build with $their_cc (the default pixi environment), or set LLVM_PREBUILT=never"
        error "to build LLVM here with the compiler you are using."
        exit 1
      fi
    fi
    exit 0
  fi
  if [[ "$llvm_prebuilt" == always ]]; then
    error "LLVM_PREBUILT=always, but no prebuilt LLVM could be used"
    exit 1
  fi
  warning "Building LLVM ${llvm_revision:0:12} from source instead (LLVM_PREBUILT=never skips the download)"
  # So that the later steps do not pick up one for another revision
  rm -rf "$llvm_prebuilt_dir"
  llvm_build_dir="$llvm_source_dir/build"
fi

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
elif [[ -f "$cache_file" ]] && cached_cxx="$(grep -E '^CMAKE_CXX_COMPILER:[A-Z]+=' "$cache_file" | sed 's/.*=//')" \
     && [[ -n "$cached_cxx" && ! "$cached_cxx" -ef "$CXX" ]]; then
  clean_reason="cached compiler ($cached_cxx) != the one we build with ($CXX)"
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
# The fork configures LLVM for Cinnamon in a CMake cache file it versions along
# with its sources, and builds the prebuilt LLVM from it too. The options below
# are only for LLVM checkouts that lack it.
llvm_config_file="$llvm_source_dir/cinnamon/llvm-config.cmake"
llvm_config_hash=""
if [[ -f "$llvm_config_file" ]]; then
  config_opts=( -C "$llvm_config_file" ${llvm_overrides[@]+"${llvm_overrides[@]}"} )
  # Cache files do not override what is already in the cache, so a change to it
  # needs a fresh build dir.
  llvm_config_hash="$(hash_cmd < "$llvm_config_file" | awk '{print $1}')"
else
  warning "No '$llvm_config_file'; configuring LLVM with the defaults of $(basename "$0")"
  config_opts=(
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_SHARED_LIBS=ON
    -DLLVM_BUILD_TOOLS=ON
    -DLLVM_ENABLE_PROJECTS="$LLVM_PROJECTS"
    -DLLVM_ENABLE_ASSERTIONS=ON
    -DLLVM_ENABLE_EH=ON
    -DLLVM_ENABLE_RTTI=ON
    -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD="$LLVM_EXPERIMENTAL_TARGETS"
    -DLLVM_INCLUDE_BENCHMARKS=OFF
    -DLLVM_INCLUDE_TESTS=OFF
    -DLLVM_INSTALL_UTILS=ON
    -DLLVM_OPTIMIZED_TABLEGEN=ON
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON
    -DLLVM_TARGETS_TO_BUILD="$LLVM_TARGETS_TO_BUILD"
  )
fi

EXTRA_CMAKE_OPTS_JOIN="${EXTRA_CMAKE_OPTS[*]}"
CURRENT_HASH="$(printf '%s\n' \
  "PROJ=$LLVM_PROJECTS" \
  "TGT=$LLVM_TARGETS_TO_BUILD" \
  "EXP=$LLVM_EXPERIMENTAL_TARGETS" \
  "OPTS=${EXTRA_CMAKE_OPTS_JOIN}" \
  "PY=${PYBIN:-}" \
  "GEN=Ninja" \
  ${llvm_config_hash:+"CFG=$llvm_config_hash"} | hash_cmd | awk '{print $1}')"

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
  "${config_opts[@]}" \
  -DLLVM_CCACHE_BUILD=ON \
  "${EXTRA_CMAKE_OPTS[@]}"

# Save config hash so we can detect future changes
echo "$CURRENT_HASH" > "$HASH_FILE"

[[ -f "$llvm_build_dir/build.ninja" ]] || { error "CMake configure did not produce build.ninja."; exit 1; }

status "Building LLVM (Ninja)"
# shellcheck disable=SC2086
cmake --build "$llvm_build_dir" --target ${LLVM_BUILD_TARGETS}

export PATH="$llvm_build_dir/bin:$PATH"

fi
