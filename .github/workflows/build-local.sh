#!/bin/bash

script_dir="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

source "$script_dir/common.sh"

source "$script_dir/setup-venv.sh"
source "$script_dir/build-llvm.sh"
source "$script_dir/build-torch.sh"
source "$script_dir/build-upmem.sh"

source "$script_dir/build-cinnamon.sh"