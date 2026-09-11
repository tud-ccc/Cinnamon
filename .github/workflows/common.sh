#!/bin/bash

if [[ -z "${PREAMBLE_LOADED:-}" ]]; then
PREAMBLE_LOADED="1"

set -e

script_dir="$( cd -- "$(dirname "${BASH_SOURCE[0]:-$0}")" >/dev/null 2>&1 ; pwd -P )"
project_root="$(realpath "$script_dir/../..")"

# Load .env file and export all variables
if [ -f "$project_root/.env" ]; then
  set -o allexport
  # shellcheck source=/dev/null
  source "$project_root/.env"
  set +o allexport
fi

status() { echo -e "\033[1m$1\033[0m"; }
info() { echo -e "\033[1;34m$1\033[0m"; }
warning() { echo -e "\033[1;33m$1\033[0m"; }
error() { echo -e "\033[1;31m$1\033[0m" >&2; }

print_and_run() {
  a="$(echo "$@")"
  status "$a"
  "$@"
}

verbose_cmd() {
  if [[ "${verbose:-0}" -eq 1 ]]; then
    "$@"
  else
    "$@" >/dev/null
  fi
}

# Check out a submodule at the revision this repository pins it to.
#
# Pass shallow=1 for large histories (LLVM). A shallow `git submodule update`
# only fetches the tip of the remote's default branch, which need not contain
# the pinned revision, so fetch that revision directly and hand the result to
# git. Submodule names match their paths throughout .gitmodules.
ensure_submodule() {
  local path="$1"
  local shallow="${2:-0}"
  local abs="$project_root/$path"
  local revision url current
  # Not `ls-tree --object-only`: that needs git 2.36.
  revision="$(git -C "$project_root" ls-tree HEAD -- "$path" | awk '$2 == "commit" { print $3 }')"

  # Presence is judged by content, not by a .git entry: CI restores these trees
  # from a cache that does not carry the corresponding .git/modules directory.
  # A lone .git does not count: that is a checkout whose files were removed.
  if [[ -n "$(ls -A "$abs" 2>/dev/null | grep -v -x '\.git')" ]]; then
    current=""
    if [[ -e "$abs/.git" ]]; then
      current="$(git -C "$abs" rev-parse -q --verify HEAD 2>/dev/null || true)"
    fi
    if [[ -n "$current" && -n "$revision" && "$current" != "$revision" ]]; then
      warning "Submodule '$path' is at ${current:0:12}, but this repository pins ${revision:0:12}."
      warning "Run 'git submodule update -- $path' unless that is intended."
    fi
    info "Submodule '$path' is already checked out"
    return 0
  fi
  if [[ -e "$abs/.git" ]]; then
    error "Submodule '$path' has its git metadata but no files."
    error "Restore them with: git -C '$abs' checkout -f ${revision:-<pinned revision>}"
    exit 1
  fi

  url="$(git -C "$project_root" config -f "$project_root/.gitmodules" --get "submodule.$path.url")"
  if [[ -z "$revision" || -z "$url" ]]; then
    error "'$path' is not a registered submodule of this repository"
    exit 1
  fi

  if [[ "$shallow" -eq 1 ]]; then
    status "Checking out submodule '$path' at $revision (shallow)"
    mkdir -p "$abs"
    git -C "$abs" init -q
    git -C "$abs" remote add origin "$url" 2>/dev/null \
      || git -C "$abs" remote set-url origin "$url"
    git -C "$abs" fetch -q --depth 1 origin "$revision"
    git -C "$abs" checkout -q FETCH_HEAD
    git -C "$project_root" submodule absorbgitdirs -- "$path"
  else
    status "Checking out submodule '$path' at $revision"
    print_and_run git -C "$project_root" submodule update --init -- "$path"
  fi
}

status "Project root: $project_root"

py_venv_path="$project_root/.venv"
cinnamon_path="$project_root"
cinnamon_build_dir="${CINNAMON_BUILD_DIR:-$project_root/build}"

verbose=0
reconfigure=0

setup_python_venv=1
build_torch_mlir=1
build_cinnamon_wheel=1

enable_cuda=0
enable_roc=0

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

if echo "$@" | grep -q -- "-no-torch-mlir"; then
  build_torch_mlir=0
fi

if echo "$@" | grep -q -- "-no-cinnamon-wheel"; then
  build_cinnamon_wheel=0
fi

if echo "$@" | grep -q -- "-enable-gpu"; then
  CINNAMON_CMAKE_OPTIONS="${CINNAMON_CMAKE_OPTIONS:-} -DCINM_BUILD_GPU_SUPPORT=ON"
fi

if echo "$@" | grep -q -- "-enable-cuda"; then
  enable_cuda=1
fi

if echo "$@" | grep -q -- "-enable-roc"; then
  enable_roc=1
fi

################################################################################
# Host compiler
#
# CMake, Conan and the sub-builds must all agree on one compiler, so resolve it
# here and export CC/CXX rather than letting each of them guess.
#
# A bare `clang` from PATH is not trustworthy: vendor SDKs ship old
# cross-compilers under that name and can sit ahead of the system one. The
# UPMEM SDK in particular provides a clang 12, which cannot compile the C++20
# this project requires. So resolve to an absolute path and check the version
# before handing it to anything.
################################################################################

gcc_min_major=12
clang_min_major=16

compiler_is_usable() {
  local cc="$1" major
  command -v "$cc" >/dev/null 2>&1 || return 1
  major="$("$cc" -dumpversion 2>/dev/null | cut -d. -f1)"
  [[ -n "$major" ]] || return 1
  if "$cc" --version 2>/dev/null | head -1 | grep -qi clang; then
    (( major >= clang_min_major ))
  else
    (( major >= gcc_min_major ))
  fi
}

describe_compiler() {
  printf '%s (%s)' "$(command -v "$1")" "$("$1" --version 2>/dev/null | head -1)"
}

if [[ -n "${CC:-}" || -n "${CXX:-}" ]]; then
  # An explicit choice is honoured, but still checked: pointing the build at a
  # compiler that cannot handle C++20 fails in a much more confusing way later.
  if [[ -z "${CC:-}" || -z "${CXX:-}" ]]; then
    error "Set both CC and CXX, or neither (CC='${CC:-}' CXX='${CXX:-}')"
    exit 1
  fi
  if ! compiler_is_usable "$CXX"; then
    error "CXX='$CXX' is not usable: $(describe_compiler "$CXX" 2>/dev/null || echo 'not found')"
    error "This project needs C++20, i.e. GCC >= $gcc_min_major or Clang >= $clang_min_major."
    exit 1
  fi
else
  for pair in "gcc-15:g++-15" "gcc-14:g++-14" "gcc-13:g++-13" "gcc-12:g++-12" \
              "clang-20:clang++-20" "clang-19:clang++-19" "clang-18:clang++-18" \
              "clang-17:clang++-17" "clang-16:clang++-16" "cc:c++"; do
    if compiler_is_usable "${pair#*:}"; then
      CC="$(command -v "${pair%%:*}")"
      CXX="$(command -v "${pair#*:}")"
      break
    fi
  done
  if [[ -z "${CXX:-}" ]]; then
    error "No C++20 host compiler found (need GCC >= $gcc_min_major or Clang >= $clang_min_major)."
    error "Install one, or set CC and CXX to the compiler you want to build with."
    if command -v clang >/dev/null 2>&1; then
      error "Note: '$(describe_compiler clang)' is on your PATH but was rejected."
    fi
    exit 1
  fi
fi

CC="$(command -v "$CC")"
CXX="$(command -v "$CXX")"
export CC CXX
info "Host compiler: $CXX ($("$CXX" --version 2>/dev/null | head -1))"

# A compiler passed through one of the *_CMAKE_OPTIONS overrides CC/CXX for
# that sub-build, so it has to pass the same check.
for opts_var in LLVM_CMAKE_OPTIONS TORCH_MLIR_CMAKE_OPTIONS CINNAMON_CMAKE_OPTIONS; do
  opts_val="${!opts_var:-}"
  [[ -n "$opts_val" ]] || continue
  # `|| true`: no match is the normal case, and grep failing must not trip
  # `set -e` / `pipefail` in the scripts that source this file.
  override="$(echo "$opts_val" | grep -o -- '-DCMAKE_CXX_COMPILER=[^ ]*' | tail -1 | cut -d= -f2- || true)"
  [[ -n "$override" ]] || continue
  if ! compiler_is_usable "$override"; then
    error "$opts_var sets -DCMAKE_CXX_COMPILER=$override, which cannot build this project:"
    error "  $(describe_compiler "$override" 2>/dev/null || echo "$override: not found")"
    error "Drop it to use the detected compiler ($CXX), or point it at a C++20 compiler."
    exit 1
  fi
done

################################################################################
# Dependency resolution
#
# Every dependency is described by three independent facts:
#
#   <dep>_source_dir   where its sources live
#   <dep>_build_dir    where its build tree lives, i.e. what CMake is pointed at
#   build_<dep>        whether we build it here, or only consume it
#
# The in-tree default for each source dir is a git submodule under third-party/.
# Setting <DEP>_SOURCE_DIR or <DEP>_BUILD_DIR points the build at a tree you
# already have elsewhere; the corresponding submodule then stays uninitialized
# and is never cloned. Downstream CMake invocations always receive explicit
# paths, whichever way they were resolved.
################################################################################

# ---- LLVM / MLIR ----
llvm_source_dir="${LLVM_SOURCE_DIR:-$project_root/third-party/llvm}"
llvm_build_dir="${LLVM_BUILD_DIR:-$llvm_source_dir/build}"
# We only build LLVM ourselves when it lives in the tree we manage.
if [[ -n "${LLVM_BUILD_DIR:-}" ]]; then
  build_llvm=0
  info "Using LLVM build tree '$llvm_build_dir' (LLVM_BUILD_DIR)"
  [[ -d "$llvm_build_dir" ]] || warning "Directory '$llvm_build_dir' does not exist"
else
  build_llvm=1
fi
if echo "$@" | grep -q -- "-no-llvm"; then
  build_llvm=0
fi
llvm_cmake_dir="$llvm_build_dir/lib/cmake/llvm"
mlir_cmake_dir="$llvm_build_dir/lib/cmake/mlir"

# ---- Torch-MLIR ----
torch_mlir_source_dir="${TORCH_MLIR_SOURCE_DIR:-$project_root/third-party/torch-mlir}"
torch_mlir_build_dir="$torch_mlir_source_dir/build"
torch_mlir_install_dir="${TORCH_MLIR_INSTALL_DIR:-$torch_mlir_source_dir/install}"
if [[ -n "${TORCH_MLIR_INSTALL_DIR:-}" ]]; then
  build_torch_mlir=0
  info "Using Torch-MLIR installation '$torch_mlir_install_dir' (TORCH_MLIR_INSTALL_DIR)"
  [[ -d "$torch_mlir_install_dir" ]] || warning "Directory '$torch_mlir_install_dir' does not exist"
fi

# ---- UPMEM SDK ----
# Not a submodule: the SDK is not publicly redistributable, so it is either
# unpacked into third-party/upmem by hand or pointed to by UPMEM_HOME.
upmem_dir="${UPMEM_HOME:-$project_root/third-party/upmem}"
enable_upmem=1
if echo "$@" | grep -q -- "-no-upmem"; then
  enable_upmem=0
fi

# ---- Python environment ----
# Activated here rather than by setup-venv.sh so that each build script is a
# self-contained program: they are run as subprocesses, not sourced, and so
# cannot inherit an environment from one another.
if [[ "$setup_python_venv" -eq 1 && -z "${VIRTUAL_ENV:-}" && -f "$py_venv_path/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$py_venv_path/bin/activate"
fi

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PYBIN="$(command -v python)"
  export LLVM_CMAKE_OPTIONS="${LLVM_CMAKE_OPTIONS:-} -DPython3_EXECUTABLE=${PYBIN} -DPython3_FIND_VIRTUALENV=ONLY"
fi

fi
