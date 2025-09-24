#!/bin/bash

script_dir="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
source "$script_dir/common.sh"

if [[ $checkout_upmem -eq 1 ]]; then
  if [ ! -d "$upmem_path" ]; then
    local_archive="$project_root/resource/upmem-2023.2.0-Linux-x86_64.tar.gz"
    if [[ -f "$local_archive" ]]; then
      status "Using bundled UpMem SDK archive"
      mkdir -p "$upmem_path"
      tar xfz "$local_archive" -C "$upmem_path" --strip-components=1
    else
      error "Bundled UpMem SDK archive not found at $local_archive"
      exit 1
    fi
  fi
elif [[ $checkout_upmem -eq 0 ]]; then
  warning "Skipping UpMem checkout"
  warning "The following steps will need UPMEM_DIR to be set in their respective <STEP>_CMAKE_OPTIONS"
fi
