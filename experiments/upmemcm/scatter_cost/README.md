This is a microbenchmark to determine cost-modelling parameters for UPMEM
host -> DPU transfer APIs, sweeping number of DPUs, blocks per DPU, and block
size. Three transfer APIs are covered, each as its own host binary built from
the same `scatter_bench.cpp` (see the `XFER_MODE` doc-comment at its top) and
sharing one DPU-side binary (`scatter_dpu.c`, whose MRAM buffer is sized for
the worst case of all three):

- `bin/scatter_bench` — `dpu_push_sg_xfer` (per-block scatter/gather; sweeps
  blocks_per_dpu 1..24).
- `bin/scatter_bench_block` — `dpu_push_xfer` (one contiguous block per DPU;
  blocks_per_dpu fixed at 1).
- `bin/scatter_bench_broadcast` — `dpu_broadcast_to` (same block copied to
  every DPU; blocks_per_dpu fixed at 1).

## Build

```
make                 # builds bin/scatter_dpu and all three bin/scatter_bench* binaries
```

## Run

```
./bin/scatter_bench             # dpu_push_sg_xfer
SCATTER_CSV_OUT=results_block.csv ./bin/scatter_bench_block
SCATTER_CSV_OUT=results_broadcast.csv ./bin/scatter_bench_broadcast
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

## Thoughts

Ok so it seems the best fit I can for this is a model that has a parameter for each of the three dimensions, one for each of their pairwise products, and one for the triple product (total bytes).

Additionally, the fit is improved if you cut the model in 2, one for low DPU counts (<=32), and another for higher DPU counts, with different parameters.

The next steps to be able to use this experiment is describe the scientific protocol. If I want to be thorough I should also do an ablation study/ a study of which parameters matter most, and prune the useless parameters. Then try to explain the trends architecturally.

I also need to do the same kind of study for broadcast, and maybe for the regular scatter and gather. The good thing is that the data I have here is very fast to collect (not 2 days like Georg's experiment) and process. But this still needs lots of work.

I also need to review the literature for existing cost models for the scatter/gather overhead. Maybe we're the first to model the UPMEM scatter API?
