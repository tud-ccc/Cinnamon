#!/usr/bin/env bash
set -euo pipefail

# Full build, in dependency order. This is what `just configure` runs.
#
# Each step is a self-contained program: it re-reads the configuration from
# common.sh rather than inheriting an environment from the previous step, so
# any one of them can also be run on its own to redo just that part.

script_dir="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

"$script_dir/setup-venv.sh" "$@"
"$script_dir/build-llvm.sh" "$@"
"$script_dir/build-torch.sh" "$@"
"$script_dir/build-upmem.sh" "$@"
"$script_dir/build-cinnamon.sh" "$@"
