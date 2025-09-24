#!/usr/bin/env bash
set -euo pipefail

# Root-level build script that runs the workflow scripts in order
# with no user-configurable flags. It builds everything, including
# ALPINE inside a Docker container (non-optional).
# Order:
#  1) Create Python venv and install deps
#  2) Source the venv
#  3) Build LLVM
#  4) Build Torch-MLIR
#  5) Download/prepare UpMem SDK
#  6) Build Cinnamon
#  7) Build ALPINE in container (required)

repo_root="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
wf_dir="$repo_root/.github/workflows"

require() {
  local p="$1"
  if [[ ! -f "$p" ]]; then
    echo "Missing required script: $p" >&2
    exit 1
  fi
}

require "$wf_dir/common.sh"
require "$wf_dir/setup-venv.sh"
require "$wf_dir/build-llvm.sh"
require "$wf_dir/build-torch.sh"
require "$wf_dir/build-cinnamon.sh"
require "$wf_dir/build-alpine.sh"
require "$wf_dir/build-upmem.sh"

# Ensure no external environment variables alter the flow
unset LLVM_BUILD_DIR TORCH_MLIR_INSTALL_DIR UPMEM_HOME \
      CINNAMON_CMAKE_OPTIONS TORCH_MLIR_CMAKE_OPTIONS LLVM_CMAKE_OPTIONS || true

# Load common to initialize defaults
# shellcheck disable=SC1091
source "$wf_dir/common.sh"

echo "==> [1/7] Setting up Python venv"
"$wf_dir/setup-venv.sh"

# Ensure the venv is active in this shell
if [[ -d "$repo_root/.venv" ]]; then
  # shellcheck disable=SC1091
  source "$repo_root/.venv/bin/activate"
fi

echo "==> [2/7] Building LLVM"
"$wf_dir/build-llvm.sh"

echo "==> [3/7] Building Torch-MLIR"
"$wf_dir/build-torch.sh"

echo "==> [4/7] Preparing UpMem SDK"
"$wf_dir/build-upmem.sh"

echo "==> [5/7] Building Cinnamon"
"$wf_dir/build-cinnamon.sh"

echo "==> [6/7] Building ALPINE (Docker container; required)"
"$wf_dir/build-alpine.sh"

echo "==> [7/7] Done."
