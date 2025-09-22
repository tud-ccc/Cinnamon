#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <path-to-aarch64-binary>" >&2
  exit 1
fi

bin_path_input="$1"
docker_image="${ALPINE_DOCKER_TAG:-alpine-gem5:latest}"

# Mount the REPO ROOT (Cinnamon), not the parent of it.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # .../Cinnamon
CFG_HOST="$ROOT/runtime/Alpine/se_alpine.py"

if [[ ! -f "$CFG_HOST" ]]; then
  echo "Missing gem5 config at $CFG_HOST" >&2
  exit 1
fi

# Resolve binary path to an absolute path.
if [[ "$bin_path_input" = /* ]]; then
  BIN_HOST="$bin_path_input"
else
  BIN_HOST="$(cd "$(dirname "$bin_path_input")" && pwd)/$(basename "$bin_path_input")"
fi

if [[ ! -f "$BIN_HOST" ]]; then
  echo "Binary not found: $BIN_HOST" >&2
  exit 1
fi

case "$BIN_HOST" in
  "$ROOT"/*) ;;
  *)
    echo "Binary must reside inside the repository tree: $ROOT" >&2
    exit 1
    ;;
esac

BIN_REL="${BIN_HOST#$ROOT/}"
BIN_C="/workspace/$BIN_REL"

BIN_DIR_HOST="$(dirname "$BIN_HOST")"
OUT_DIR_HOST="$BIN_DIR_HOST/out"
OUT_REL="${OUT_DIR_HOST#$ROOT/}"
OUT_C="/workspace/$OUT_REL"

# Container paths (hardcoded)
GEM5_BIN_C=/workspace/third-party/ALPINE/gem5-X-ALPINE/build/ARM/gem5.opt
SE_CFG_C=/workspace/runtime/Alpine/se_alpine.py
PYTHONPATH_C=/workspace/third-party/ALPINE/gem5-X-ALPINE/configs

# Create out folder only if it's missing
if [[ ! -d "$OUT_DIR_HOST" ]]; then
  mkdir -p "$OUT_DIR_HOST"
fi

echo "[host] PWD: $(pwd)"
echo "[host] ROOT: $ROOT"
echo "[host] docker image: $docker_image"
echo "[host] binary: $BIN_HOST"
echo "[host] config: $CFG_HOST"
echo "[host] output dir: $OUT_DIR_HOST"
docker run --rm \
  -u "$(id -u):$(id -g)" \
  -v "$ROOT":/workspace \
  -w /workspace \
  -e PYTHONPATH="$PYTHONPATH_C" \
  "$docker_image" /bin/sh -eu -c "
    echo '[docker] PWD:' \\$(pwd)
    echo '[docker] Exec: $GEM5_BIN_C $SE_CFG_C -c $BIN_C'
    exec $GEM5_BIN_C $SE_CFG_C \
      --arm-iset aarch64 \
      --cpu-type=AtomicSimpleCPU -n 1 --fastmem \
      --mem-type=SimpleMemory --mem-size=2GB \
      --output=$OUT_C/program.out \
      -c $BIN_C
  "
