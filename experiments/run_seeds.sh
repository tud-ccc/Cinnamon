#!/usr/bin/env bash
# Run cinm-opt for N seeds in parallel, limited to nproc workers.
#
# Usage: run_seeds.sh <file> <dir> <scale> <n_seeds> [extra infer-accelerator opts...]
#
# Each seed writes:
#   data/<dir>/<file>_seed<N>.log   — stderr (cinm-opt diagnostics)
#   data/<dir>/out_seed<N>.mlir     — stdout (IR output)
set -euo pipefail

FILE="$1";  shift
DIR="$1";   shift
SCALE="$1"; shift
N="$1";     shift
EXTRA="${*}"   # remaining args forwarded into --upmem-infer-accelerator

MAX_JOBS=$(nproc)
mkdir -p "data/$DIR"

pids=()
failed=0

# Wait until fewer than MAX_JOBS background processes are running.
drain_to_limit() {
    while (( ${#pids[@]} >= MAX_JOBS )); do
        wait -n 2>/dev/null || true
        live=()
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                live+=("$pid")
            else
                # Harvest exit status without blocking (already done)
                wait "$pid" 2>/dev/null || (( failed++ )) || true
            fi
        done
        pids=("${live[@]}")
    done
}

for seed in $(seq 1 "$N"); do
    drain_to_limit
    INFER_OPTS="${EXTRA:+$EXTRA }objective-scale=$SCALE dump-dir=data/$DIR rng-seed=$seed"
    cinm-opt "$FILE.mlir" \
        --cinm-assign-platforms \
        --cinm-isolate-compute-blocks "--upmem-infer-accelerator=$INFER_OPTS" \
        --mlir-print-ir-after-failure --dump-pass-pipeline \
        --split-input-file --debug-only=cinm-inference \
        2> /dev/null \
        > /dev/null &
    pids+=($!)
    echo "Started seed $seed (pid $!)"
done

for pid in "${pids[@]}"; do
    wait "$pid" || (( failed++ )) || true
done

if (( failed > 0 )); then
    echo "WARNING: $failed seed(s) failed — check the log files in data/$DIR/" >&2
    exit 1
fi
