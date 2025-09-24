#!/usr/bin/env bash
set -euo pipefail

repo_root="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

if [[ "$(uname -s)" == "Darwin" ]]; then
  llvm_lib_dir="$repo_root/third-party/llvm/build/lib"
  llvm_cmake_dir="$repo_root/third-party/llvm/build/lib/cmake/llvm"
  if [[ -d "$llvm_lib_dir" ]]; then
    export DYLD_LIBRARY_PATH="$llvm_lib_dir${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
  fi
  if [[ -d "$llvm_cmake_dir" ]]; then
    export LLVM_DIR="$llvm_cmake_dir"
  fi
fi

if [[ -d "$repo_root/.venv" ]]; then
  # shellcheck disable=SC1091
  source "$repo_root/.venv/bin/activate"
fi

if command -v jupyter >/dev/null 2>&1; then
  exec jupyter notebook --notebook-dir="$repo_root"
fi

python="$(command -v python3 || command -v python || true)"
if [[ -z "$python" ]]; then
  echo "Cannot find 'jupyter' or a Python interpreter to launch it." >&2
  exit 1
fi

if ! "$python" - <<'PY' >/dev/null 2>&1
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("notebook") else 1)
PY
then
  echo "Installing notebook into the active Python environment..." >&2
  "$python" -m pip install --quiet notebook
fi

exec "$python" -m notebook --notebook-dir="$repo_root"
