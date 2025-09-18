#!/usr/bin/env bash
set -euo pipefail

script_dir="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
source "$script_dir/common.sh"

# ---- Safe defaults (avoid 'unbound variable') ----
setup_python_venv="${setup_python_venv:-0}"
reconfigure="${reconfigure:-0}"
enable_cuda="${enable_cuda:-0}"
enable_roc="${enable_roc:-0}"
py_venv_path="${py_venv_path:?Define 'py_venv_path' in common.sh}"

if [[ "$setup_python_venv" -eq 1 ]]; then
  # Prefer a Python version that plays well with PyTorch wheels on many distros.
  supported_python_executable="python3"
  for cand in python3.12 python3.11 python3.10; do
    if command -v "$cand" >/dev/null 2>&1; then
      supported_python_executable="$cand"
      break
    fi
  done

  reconfigure_python_venv=0
  if [[ ! -d "$py_venv_path" ]]; then
    status "Creating Python venv ($supported_python_executable)"
    if ! "$supported_python_executable" -m venv "$py_venv_path"; then
      error "Cannot create venv at $py_venv_path"
      exit 1
    fi
    # shellcheck disable=SC1091
    source "$py_venv_path/bin/activate"
    reconfigure_python_venv=1
  else
    status "Enabling Python venv"
    # shellcheck disable=SC1091
    source "$py_venv_path/bin/activate"
  fi

  # Determine which PyTorch index to use
  if [[ "$enable_cuda" -eq 1 && "$enable_roc" -eq 1 ]]; then
    warning "Both enable_cuda and enable_roc are set; defaulting to CUDA wheels."
  fi

  if   [[ "$enable_cuda" -eq 1 ]]; then torch_source="https://download.pytorch.org/whl/cu124"
  elif [[ "$enable_roc"  -eq 1 ]]; then torch_source="https://download.pytorch.org/whl/rocm6.1"
  else                                torch_source="https://download.pytorch.org/whl/cpu"
  fi

  if [[ "$reconfigure" -eq 1 || "$reconfigure_python_venv" -eq 1 ]]; then
    status "Installing Python dependencies into venv"
    verbose_cmd python -m pip install --upgrade pip
    # PyTorch first (per official guidance), then build tooling & bindings
    verbose_cmd pip install torch torchvision torchaudio --index-url "$torch_source"
    # Ensure pybind11 >= 2.10 for MLIR, plus numpy, nanobind, build
    verbose_cmd pip install "pybind11>=2.10" numpy nanobind build
  fi

  # Ensure CMake will use this venv's Python and find pybind11's CMake config
  PYBIN="$(command -v python)"
  if ! PYBIND11_DIR="$("$PYBIN" - <<'PY'
import sys
try:
    import pybind11
    print(pybind11.get_cmake_dir())
except Exception as e:
    sys.exit(1)
PY
)"; then
    # If import failed (e.g., user skipped reconfigure), install pybind11 now.
    status "pybind11 not found in venv; installing..."
    verbose_cmd pip install "pybind11>=2.10"
    PYBIND11_DIR="$("$PYBIN" -c 'import pybind11; print(pybind11.get_cmake_dir())')"
  fi

  # Export extra CMake flags for downstream scripts (LLVM/MLIR configure)
  # - Use the venv Python
  # - Point CMake to pybind11's CMake package dir
  # - And force CMake to prefer the active virtualenv
  export LLVM_CMAKE_OPTIONS="${LLVM_CMAKE_OPTIONS:-} -DPython3_EXECUTABLE=${PYBIN} -Dpybind11_DIR=${PYBIND11_DIR} -DPython3_FIND_VIRTUALENV=ONLY"

elif [[ "$setup_python_venv" -eq 0 ]]; then
  warning "Skipping Python venv setup"
  warning "Make sure your active Python has compatible torch, numpy, and pybind11 (>=2.10)."
fi
