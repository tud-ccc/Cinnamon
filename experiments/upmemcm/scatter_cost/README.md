This is a microbenchmark to determine cost-modelling parameters for
`dpu_push_sg_xfer` scatter transfers (host -> DPU), sweeping number of DPUs,
blocks per DPU, and block size.

## Build

```
make                 # builds bin/scatter_dpu and bin/scatter_bench
```

## Run

```
./bin/scatter_bench
```

Sweeps a reduced default range (~4.8k configs, ~1 minute) and appends results
incrementally to `results.csv` as it goes, so an interrupted run still leaves
usable data. Env vars:

- `SCATTER_DENSE=1` — sweep the full originally-requested ranges (num DPUs
  1-2048 taking 1-10/powers-of-2/multiples-of-32, block sizes 8-8192 taking
  powers-of-2/multiples-of-32; ~464k configs, expect a multi-hour run —
  intended to be started separately, e.g. under `tmux`, once the default
  sweep above has confirmed the harness works).
- `SCATTER_ITERS` (default 3), `SCATTER_WARMUP` (default 1) — timed/untimed
  repetitions per config.
- `SCATTER_CSV_OUT` (default `results.csv`) — output path.

Blocks are laid out non-contiguously in host memory (padded so no two blocks
are adjacent), forcing genuine per-block scatter dispatch rather than letting
the SDK coalesce adjacent blocks into one contiguous copy.

## Analyze

```
python3 analyze.py results.csv
```

Prints a table comparing several regression templates (raw dims, log2(block
size), log2 of everything, total_bytes, ...) fit by OLS, and writes
latency-vs-block-size / latency-vs-num-dpus / blocks-vs-size-heatmap /
regression-fit plots to `plots/`.
