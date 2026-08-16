# Manually-authored configuration points

One JSON file per benchmark and source, checked in and reviewed like code:

- `atim_published/{bench}.json` — the schedule ATiM ships with its
  artifact, transcribed from its trace file into our parameter space (read
  the per-parameter `doc` strings and the permutation tables in that
  benchmark's `space.json`; every dimension the space declares must be
  named).
- `atim_reproduced/{bench}.json` — the schedule ATiM's own autotuner found
  when we ran it on this machine, transcribed the same way.
- `cinm1rule/{bench}.json` — the configuration CINM 1.0's rule-based
  tiling decides for the same function.

The two ATiM sources are kept apart everywhere downstream, not merged into
one "ATiM". They come from different searches on different hardware, they
transcribe to different points, and the better of the two is a
configuration neither run produced. E1 decomposes the gap to a *specific*
ATiM configuration, so which one it was has to stay on the row.

File format — a list of points, one per function:

```json
[
  {
    "fn_name": "gemv_64MB",
    "trace": "where this came from (ATiM trace path / rule invocation)",
    "candidates": [
      {
        "label": "r0",
        "params": {"dpus": 2048, "tasklets": 8, "gemv.M.mram": 8, "...": 0},
        "expected": {
          "D": [2048],
          "T": [8],
          "scattered_bytes": 67108864,
          "wram_bytes_per_tasklet": 2560
        }
      }
    ]
  }
]
```

Several `candidates` per point is the intended answer to transcription
ambiguity: when the trace admits more than one reading, transcribe every
reading and measure them all rather than guessing.

`expected` is optional and holds whatever invariants the transcriber
derived from the trace by hand (`D`, `T` as lists — one entry per DPU set —
plus `scattered_bytes`, `gathered_bytes`, `wram_bytes_per_tasklet`).
`doit invariants_report` recomputes these from the compiled artifact and
flags disagreements, so a mistranscription is caught before it costs
hardware time. `doit compile_points bench_points` measures the candidates
through the exact same pipeline as every other configuration.

## Transcribing an ATiM trace

Per benchmark, per trace file (e.g. ATiM's
`evaluation/results/tuned_modules/atim_gemv_1024_1_1024.py`):

1. Read the `apply_trace_*` function, not the lowered module: the
   `sample_perfect_tile(..., decision=[...])` lines carry every tiling
   factor; the `bind(... thread_axis="blockIdx.*")` rows give D (product
   of the bank axes) and `threadIdx.x` gives T; `sch.reorder(...)`
   together with the split structure gives the loop order; `rfactor` is
   a host-side reduction split (K-split).
2. Open the benchmark's `data/{bench}/space/space.json` (`doit space`)
   and map each decision onto our named parameters using the per-param
   `doc` strings. For an order parameter, look the loop order up in the
   listed permutation table and copy its encoding verbatim: one
   `name=value` pair per *dimension*; a permutation of n items occupies
   dimensions `name[0]..name[n-1]`, each holding the 1-based place of
   item k; there is one order parameter per op, and ops with a single
   iteration dimension have none. A missing or unknown name is a loud
   error listing the expected set, and an infeasible value is rejected
   by feasible-set membership — a wrong guess cannot slip through
   silently.
3. Write the `{bench}.json`; when the trace admits several readings,
   write them all as candidates.
4. `doit compile_points invariants_report` — agreement on every
   cost-determining invariant is the structural validation the paper
   cites; a disagreement names the axis that was misread.

The same procedure fills `cinm1rule/`: run the CINM 1.0 pipeline once
per benchmark, read the tiling factors its rule infers, and transcribe
them into the MRAM-tiling-disabled subspace.
