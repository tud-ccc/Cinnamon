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

# Torch-MLIR builds on MLIR's Python bindings, which are built for one Python
# version. Ours has to be that version, or what we install below cannot import
# them, with an ImportError far from its cause.
# -print -quit, not `| head -1`: that kills find with SIGPIPE, which pipefail
# then turns into the death of this script.
mlir_pyext="$(find "$llvm_build_dir" -path '*/mlir/_mlir_libs/*' -name '*.cpython-*.so' -print -quit 2>/dev/null || true)"
if [[ -n "$mlir_pyext" ]]; then
  mlir_pytag="$(basename "$mlir_pyext" | sed -n 's/.*\.\(cpython-[0-9]\+\)-.*/\1/p')"
  our_pytag="$("$python_for_install" -c 'import sysconfig; print("-".join(sysconfig.get_config_var("SOABI").split("-")[:2]))' 2>/dev/null || true)"
  if [[ -n "$mlir_pytag" && -n "$our_pytag" && "$mlir_pytag" != "$our_pytag" ]]; then
    error "The MLIR Python bindings in '$llvm_build_dir' are built for $mlir_pytag,"
    error "but this build uses $our_pytag ($python_for_install)."
    error "Build that LLVM with this Python, use one whose bindings match, or pass -no-torch-mlir."
    exit 1
  fi
fi

cache_file="$torch_mlir_build_dir/CMakeCache.txt"
need_config=0
cached_llvm_dir="$(grep -E '^LLVM_DIR:[A-Z]+=' "$cache_file" 2>/dev/null | sed 's/.*=//' || true)"
cached_cxx="$(grep -E '^CMAKE_CXX_COMPILER:[A-Z]+=' "$cache_file" 2>/dev/null | sed 's/.*=//' || true)"

if [[ ! -f "$cache_file" ]]; then
  need_config=1
elif ! grep -q 'CMAKE_GENERATOR:INTERNAL=Ninja' "$cache_file"; then
  status "Existing Torch-MLIR build dir is not Ninja -> recreating it"
  rm -rf "$torch_mlir_build_dir"
  need_config=1
elif [[ ! -f "$torch_mlir_build_dir/build.ninja" ]]; then
  # A configure that failed leaves a cache behind but no build files.
  need_config=1
elif [[ -n "$cached_cxx" && ! "$cached_cxx" -ef "$CXX" ]]; then
  # CMake will not change the compiler of a build tree, and the cached one may
  # not even exist any more: an LLVM build it came from could have been
  # reconfigured without the clang project, or removed.
  status "Torch-MLIR was built with '$cached_cxx', not '$CXX' -> recreating its build dir"
  rm -rf "$torch_mlir_build_dir"
  need_config=1
elif [[ -n "$cached_llvm_dir" && ! "$cached_llvm_dir" -ef "$llvm_cmake_dir" ]]; then
  # Its binaries have the old LLVM's lib directory baked in as their RPATH.
  status "Torch-MLIR was built against the LLVM in '$cached_llvm_dir' -> recreating its build dir"
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
  # Without this, CMake takes the highest Python version it can find, which is
  # the system one on a distribution that ships a newer Python than ours. Its
  # MLIR bindings and nanobind are then missing, and it does not match the
  # interpreter we install the package into below.
  if [[ -n "${PYBIN:-}" ]]; then
    dependency_paths+=( -DPython3_EXECUTABLE="$PYBIN" )
  fi

  llvm_lib_dir="$llvm_build_dir/lib"
  case "$(uname -s)" in
    Darwin) linker_flags="-L${llvm_lib_dir} -Wl,-rpath,${llvm_lib_dir} -lMLIRParser" ;;
    *)      linker_flags="-Wl,--no-as-needed -L${llvm_lib_dir} -Wl,-rpath,${llvm_lib_dir} -lMLIRParser" ;;
  esac
  # These replace the flags CMake takes from LDFLAGS, so carry them over.
  # Not --as-needed though, which conda's clang passes: Torch-MLIR links
  # libMLIRCastInterfaces before the archive that needs it, and --as-needed
  # drops a library that nothing needs *yet*, leaving undefined references to
  # mlir::impl::foldCastInterfaceOp and friends.
  linker_flags="${linker_flags}${LDFLAGS:+ ${LDFLAGS//-Wl,--as-needed/-Wl,--no-as-needed}}"
  export LDFLAGS="${LDFLAGS//-Wl,--as-needed/-Wl,--no-as-needed}"

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
    -DMLIR_BINDINGS_PYTHON_NB_DOMAIN=mlir \
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
