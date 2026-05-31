#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_CLEANER="$ROOT_DIR/tutorial/clean_notebooks.py"

if [ ! -f "$PYTHON_CLEANER" ]; then
  echo "error: Python cleaner not found at $PYTHON_CLEANER" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "error: python3 is required but was not found in PATH" >&2
  exit 1
fi

exec python3 "$PYTHON_CLEANER" "$@"
