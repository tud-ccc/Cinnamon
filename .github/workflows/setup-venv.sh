#!/bin/bash

source "common.sh"

if [[ $setup_python_venv -eq 1 ]]; then
  # NOTE: This is a temporary workaround as some distros ship python3.13 which does not yet provide a torch package
  supported_python_executable=python3
  if command -v python3.12 &> /dev/null; then
    supported_python_executable=python3.12
  fi

  reconfigure_python_venv=0
  if [ ! -d "$py_venv_path" ]; then
    status "Creating Python venv"
    
    if ! $supported_python_executable -m venv "$py_venv_path"; then
      echo "Error: cannot create venv"
      exit 1
    fi
    source "$py_venv_path/bin/activate"
    reconfigure_python_venv=1
  else
    status "Enabling Python venv"
    source "$py_venv_path/bin/activate"
  fi

  if [ $reconfigure -eq 1 ] || [ $reconfigure_python_venv -eq 1 ]; then
    status "Installing Python dependencies"
    # https://pytorch.org/get-started/locally/
    if [[ $enable_cuda -eq 1 ]]; then
      torch_source=https://download.pytorch.org/whl/cu124
    elif [[ $enable_roc -eq 1 ]]; then
      torch_source=https://download.pytorch.org/whl/rocm6.1
    else
      torch_source=https://download.pytorch.org/whl/cpu
    fi

    verbose_cmd pip install --upgrade pip
    verbose_cmd pip install torch torchvision torchaudio --index-url $torch_source
    verbose_cmd pip install pybind11 nanobind build numpy 
  fi
elif [[ $setup_python_venv -eq 0 ]]; then
  warning "Skipping Python venv setup"
  warning "Make sure to have a correct Python environment set up"
fi
