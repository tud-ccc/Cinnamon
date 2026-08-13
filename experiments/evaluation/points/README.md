# Manually-authored configuration points

One JSON file per benchmark and source, checked in and reviewed like code:

- `atim/{bench}.json` — ATiM's tuned schedule, transcribed from its trace
  file into our parameter space (read the per-parameter `doc` strings and
  the permutation tables in that benchmark's `space.json`; every dimension
  the space declares must be named).
- `cinm1rule/{bench}.json` — the configuration CINM 1.0's rule-based
  tiling decides for the same function.

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
