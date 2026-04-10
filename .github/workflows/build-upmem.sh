#!/bin/bash

script_dir="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
source "$script_dir/common.sh"

if [[ $checkout_upmem -eq 1 ]]; then
  if [ ! -d "$upmem_path" ]; then
    warning "Upmem SDK cannot be found. Unpack it at $upmem_path or set UPMEM_DIR in the cmake options to use another path."
    warning "Keep in mind the SDK is not publicly available anymore, you need to have downloaded it before it went offline."
  fi
elif [[ $checkout_upmem -eq 0 ]]; then
  warning "Skipping Upmem checkout"
  warning "The following steps will need UPMEM_DIR to be set in their respective <STEP>_CMAKE_OPTIONS"
fi
