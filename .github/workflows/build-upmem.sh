#!/usr/bin/env bash
set -euo pipefail

script_dir="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
# shellcheck source=/dev/null
source "$script_dir/common.sh"

if [[ "$enable_upmem" -eq 0 ]]; then
  warning "Skipping UPMEM SDK; the UPMEM runtime library will not be built"
  exit 0
fi

if [[ ! -d "$upmem_dir" ]]; then
  warning "UPMEM SDK not found at '$upmem_dir'."
  warning "Unpack it there or set UPMEM_HOME to another path, or pass -no-upmem."
  warning "Keep in mind the SDK is not publicly available anymore, you need to have downloaded it before it went offline."
fi
