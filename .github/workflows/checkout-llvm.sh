#!/usr/bin/env bash
set -euo pipefail

script_dir="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
# shellcheck source=/dev/null
source "$script_dir/common.sh"

# ---- Safe defaults ----
checkout_and_build_llvm="${checkout_and_build_llvm:-1}"
llvm_path="${llvm_path:?Define 'llvm_path' in common.sh}"

# ---- Clone if missing (only when requested) ----
if [[ ! -d "$llvm_path" ]]; then
  if [[ "$checkout_and_build_llvm" -eq 1 ]]; then
    status "Checking out LLVM"
    print_and_run git clone https://github.com/oowekyala/llvm-project.git --depth 1 --branch tilefirst-llvm "$llvm_path"
  else
    error "LLVM path '$llvm_path' does not exist. Set checkout_and_build_llvm=1 to clone, or create it manually."
    exit 1
  fi
else
  status "Found existing LLVM at: $llvm_path"
fi
