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

# LLVM wires up ccache itself via LLVM_CCACHE_BUILD; our own build has to ask.
ccache_opts=()
if command -v ccache >/dev/null 2>&1; then
  ccache_opts=(
    -DCMAKE_C_COMPILER_LAUNCHER=ccache
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
  )
fi

# ---- If a venv is active, make CMake use its Python ----
python_opts=()
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  python_opts=( -DPython3_EXECUTABLE="$PYBIN" )
fi

# ---- Decide whether we need to configure ----
need_config=0
reason=""
cached_llvm_dir="$(grep -E '^LLVM_DIR:[A-Z]+=' "$cache_file" 2>/dev/null | sed 's/.*=//' || true)"
cached_torch_mlir_dir="$(grep -E '^TORCH_MLIR_DIR:[A-Z]+=' "$cache_file" 2>/dev/null | sed 's/.*=//' || true)"
if [[ ! -f "$cache_file" ]]; then
  reason="no CMake cache in '$cinnamon_build_dir'"
elif ! grep -q 'CMAKE_GENERATOR:INTERNAL=Ninja' "$cache_file"; then
  reason="existing build is not Ninja"
elif [[ ! -f "$cinnamon_build_dir/build.ninja" ]]; then
  reason="build.ninja missing"
elif [[ -n "$cached_llvm_dir" && ! "$cached_llvm_dir" -ef "$llvm_cmake_dir" ]]; then
  reason="LLVM moved from '$cached_llvm_dir' to '$llvm_cmake_dir'"
elif [[ -d "$torch_mlir_install_dir" && ! "$cached_torch_mlir_dir" -ef "$torch_mlir_install_dir" ]]; then
  # Including a tree configured before there was one to find
  reason="the Torch-MLIR installation is now '$torch_mlir_install_dir'"
elif [[ "$reconfigure" -eq 1 ]]; then
  reason="forced reconfigure (reconfigure=1)"
elif [[ -n "${PYBIN:-}" ]]; then
  cached_py="$(grep -E '^Python3_EXECUTABLE:FILEPATH=' "$cache_file" | sed 's/.*=//' || true)"
  if [[ -n "$cached_py" && "$cached_py" != "$PYBIN" ]]; then
    reason="cached Python ($cached_py) != venv Python ($PYBIN)"
  fi
fi
[[ -n "$reason" ]] && need_config=1

cached_cxx="$(grep -E '^CMAKE_CXX_COMPILER:[A-Z]+=' "$cache_file" 2>/dev/null | sed 's/.*=//' || true)"
if [[ -n "$cached_cxx" && ! "$cached_cxx" -ef "$CXX" ]]; then
  # CMake will not change the compiler of a build tree, so this one has to go.
  warning "Cinnamon was built with '$cached_cxx', now building with '$CXX'"
  warning "Recreating '$cinnamon_build_dir'"
  # Except for the Torch-MLIR installation, which build-torch.sh has just
  # refreshed in there and which the configure below is pointed at.
  for entry in "$cinnamon_build_dir"/{*,.[!.]*}; do
    [[ "$entry" -ef "$torch_mlir_install_dir" ]] || rm -rf "$entry"
  done
  reason="the compiler changed"
  need_config=1
fi

BUILD_TYPE="${CMAKE_BUILD_TYPE:-RelWithDebInfo}"

if [[ "$need_config" -eq 1 ]]; then
  status "Configuring Cinnamon (Ninja): $reason"

  # ---- Conan: install C++ dependencies into the build dir ----
  if ! command -v conan >/dev/null 2>&1; then
    error "conan not found. Run setup-venv.sh, or install conan into the active environment."
    exit 1
  fi
  mkdir -p "$cinnamon_build_dir"

  # Our C++ dependencies must be built with the same compiler and standard
  # library as Cinnamon itself, so the profile follows the resolved host
  # compiler instead of being checked in with a hardcoded one.
  conan_profile="$cinnamon_build_dir/conan-profile"
  if "$CXX" --version 2>/dev/null | head -1 | grep -qi clang; then
    conan_compiler=clang
  else
    conan_compiler=gcc
  fi
  case "$(uname -m)" in
    aarch64|arm64) conan_arch=armv8 ;;
    *)             conan_arch="$(uname -m)" ;;
  esac
  if [[ "$(uname -s)" == "Darwin" ]]; then
    conan_os=Macos
    conan_libcxx=libc++
  else
    conan_os=Linux
    conan_libcxx=libstdc++11
  fi
  cat > "$conan_profile" <<EOF
[settings]
arch=$conan_arch
build_type=$BUILD_TYPE
compiler=$conan_compiler
compiler.cppstd=20
compiler.libcxx=$conan_libcxx
compiler.version=$("$CXX" -dumpversion | cut -d. -f1)
os=$conan_os

[buildenv]
CC=$CC
CXX=$CXX

[conf]
tools.cmake.cmaketoolchain:generator=Ninja
tools.build:compiler_executables={"c": "$CC", "cpp": "$CXX"}
EOF
  status "Running conan install"
  verbose_cmd conan install . --output-folder="$cinnamon_build_dir" --build=missing \
    -s build_type="$BUILD_TYPE" -pr:h "$conan_profile" -pr:b "$conan_profile"

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
    -DCMAKE_DISABLE_PRECOMPILE_HEADERS=ON \
    "${dep_opts[@]}" \
    ${ccache_opts[@]+"${ccache_opts[@]}"} \
    ${python_opts[@]+"${python_opts[@]}"} \
    ${user_opts[@]+"${user_opts[@]}"}
  # shellcheck disable=SC1091
  source "$cinnamon_build_dir/deactivate_conanbuild.sh"
fi

# Link the LLVM tools next to ours, so that the opt, llc and mlir-translate on
# PATH are the ones of the LLVM this build uses, whether that is the prebuilt
# one, the submodule or LLVM_BUILD_DIR. IR from one version of LLVM is not
# necessarily readable by the tools of another.
#
# llvm-symbolizer is here for a second reason: LLVM symbolizes the backtrace of
# a crash with the one it finds next to the crashing executable, or on PATH.
llvm_tools=( llvm-symbolizer opt llc llvm-as llvm-dis llvm-link mlir-opt mlir-translate )
mkdir -p "$cinnamon_build_dir/bin"
for tool in "${llvm_tools[@]}"; do
  if [[ -x "$llvm_build_dir/bin/$tool" ]]; then
    ln -sfn "$llvm_build_dir/bin/$tool" "$cinnamon_build_dir/bin/$tool"
  elif [[ -L "$cinnamon_build_dir/bin/$tool" ]]; then
    # The LLVM we link to no longer has it
    rm -f "$cinnamon_build_dir/bin/$tool"
  fi
done

status "Building Cinnamon (Ninja)"
# shellcheck disable=SC2086
print_and_run cmake --build "$cinnamon_build_dir" --target all $CINNAMON_BUILD_OPTIONS

# ---- Python package wiring ----
if [[ "$setup_python_venv" -eq 1 ]]; then
  status "Building Cinnamon Python package"
  site_packages_dir="$(python -c 'import sysconfig; p=sysconfig.get_paths(); print(p.get("platlib") or p.get("purelib"))')"
  cinnamon_python_package_dir_src="$project_root/python/cinnamon/src/cinnamon"
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
    pushd "$cinnamon_path/python/cinnamon" >/dev/null
    PYTHONWARNINGS=ignore verbose_cmd python -m build
    popd >/dev/null
  fi
else
  warning "Skipping Cinnamon Python package build"
  warning "Ensure your Python env is set up if you need it."
fi
