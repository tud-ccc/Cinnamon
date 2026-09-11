#!/usr/bin/env bash
set -euo pipefail

script_dir="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
# shellcheck source=/dev/null
source "$script_dir/common.sh"

if [[ "$setup_python_venv" -eq 0 ]]; then
  warning "Skipping Python venv setup"
  warning "Make sure your active Python has compatible torch, numpy and nanobind."
  exit 0
fi

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
  reconfigure_python_venv=1
  # common.sh could not activate a venv that did not exist yet.
  # shellcheck disable=SC1091
  source "$py_venv_path/bin/activate"
fi

# Determine which PyTorch index to use
if [[ "$enable_cuda" -eq 1 && "$enable_roc" -eq 1 ]]; then
  warning "Both enable_cuda and enable_roc are set; defaulting to CUDA wheels."
fi

if   [[ "$enable_cuda" -eq 1 ]]; then torch_source="https://download.pytorch.org/whl/cu124"
elif [[ "$enable_roc"  -eq 1 ]]; then torch_source="https://download.pytorch.org/whl/rocm6.1"
else                                  torch_source="https://download.pytorch.org/whl/cpu"
fi

if [[ "$reconfigure" -eq 1 || "$reconfigure_python_venv" -eq 1 ]]; then
  status "Installing Python dependencies into venv"
  verbose_cmd python -m pip install --upgrade pip
  verbose_cmd python -m pip install "cmake==4.2.1"
  # PyTorch first (per official guidance), then build tooling & bindings
  verbose_cmd pip install torch torchvision torchaudio --index-url "$torch_source"
  verbose_cmd pip install build wheel conan

  # MLIR's Python bindings pin their own dependencies. The list lives in the
  # LLVM sources, so it needs those checked out even when we don't build LLVM.
  if [[ "$build_llvm" -eq 1 && -z "${LLVM_SOURCE_DIR:-}" ]]; then
    ensure_submodule third-party/llvm 1
  fi
  mlir_requirements="$llvm_source_dir/mlir/python/requirements.txt"
  if [[ -f "$mlir_requirements" ]]; then
    verbose_cmd python -m pip install -r "$mlir_requirements"
  else
    warning "No MLIR Python requirements at '$mlir_requirements'"
    warning "Set LLVM_SOURCE_DIR to your LLVM checkout if the bindings fail to import."
  fi

  verbose_cmd python -m pip install -r "$project_root/experiments/requirements.txt"

  # Neither package has an upstream Conan recipe; ours live in-tree.
  verbose_cmd conan export "$project_root/third-party/conan-recipes/mlpack" --name mlpack --version 4.8.0
  verbose_cmd conan export "$project_root/third-party/conan-recipes/gecode" --name gecode --version 6.4.0
fi

# MLIR's Python bindings, which Torch-MLIR builds on, need nanobind. It comes
# from MLIR's own requirements.txt, installed above.
if ! python -c 'import nanobind' 2>/dev/null; then
  error "nanobind not found in venv; run with -reconfigure, or set LLVM_SOURCE_DIR so that MLIR's Python requirements are found"
  exit 1
fi
status "Python environment ready at $py_venv_path"
