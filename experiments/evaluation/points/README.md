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

## Generating the ATiM points

`python points/transcribe_atim.py` writes both `atim_*/` directories from
the `*.tir.py` dumps in `points/traces/`. It reads the scheduled TIR rather
than the `apply_trace_*` decision list, because the TIR already states what
the decisions imply: thread-binding extents give D and T, `T.axis.spatial`
gives each dimension's extent and the loops indexing it, and the `*_local`
staging blocks give the WRAM tile as their own loop bounds.

It generates; it does not certify. Run `doit compile_points
invariants_report` afterwards and fix the JSON by hand where either
disagrees — that is the workflow, not a fallback. The script prints what it
could not derive, and checks the arithmetic it can before writing (tiles
divide, each op's leaves fill the workgroup), so a point that cannot exist
says so without costing a compile.

Where it cannot know, it emits candidates instead of guessing: ATiM's TIR
says a dimension sits on `blockIdx.x` or `blockIdx.y` but not which is the
slower-varying DPU index, so every reading of the order is measured. That
is why a three-dimensional benchmark yields six candidates.

The hand procedure below is what the script automates. It is the reference
for reading a trace when a generated point has to be corrected.

## Checking a transcription against ATiM's kernel

`python points/dump_atim_c.py` writes `points/traces/{stem}.dpu.c` beside
every `{stem}.tir.py`: the DPU source ATiM's own UPMEM backend generates for
that schedule, obtained by parsing the dump back into its module and calling
`tvm.build` on it the way `evaluation/base.py:pre_kernel` does.

That side-by-side is what decides whether a candidate is faithful, because
the invariants a point declares do not separate the candidates. Two readings
of the same trace can agree on D, T and every tiling factor and still bind a
different loop to the tasklet axis; what that changes is which operand ends
up replicated per tasklet, and that is legible only in the per-DPU buffer
declarations at the top of each kernel. Line the two `.dpu.c` files up and
compare the buffer sizes first, then the index expressions of each
`mram_read`: an operand ATiM indexes without a `tasklet_id` term is shared
across the DPU's tasklets, and a transcription that indexes it with one is
sending the same bytes several times over.

It needs ATiM's TVM and the UPMEM SDK, not DPUs. The script re-execs itself
under an interpreter that can import that TVM (`--python`, `$ATIM_PYTHON`,
then ATiM's `atim-venv`), since the evaluation's own venv cannot — NumPy 2.0
removed the aliases TVM's ctypes layer reads at import. Point it at the
checkout with `--atim` or `$ATIM_HOME`, and export `UPMEM_HOME`.

Do not read ATiM's `logs/results_*/<lambda>_0/upmem.c` instead. It is
whatever schedule that tuning run built last, carries no record of which,
and is not necessarily the incumbent that was measured.

## What ATiM's transfers cost, and what it excludes

`python points/atim_transfer_report.py` reports, per operand of every
trace, how many `dpu_push_xfer` calls its transfer is broken into and — from
ATiM's own `static_h2d.csv` — what that cost and the bandwidth it reached.
`--csv` writes the same table for downstream use.

Bytes do not predict the cost; push count does. `ExtractPimTransferSchedule`
collapses an operand into a single push only when each DPU's MRAM image is a
contiguous slice of the host buffer. Where the schedule permutes the host
and device index order the DPU image is a transposed view, no contiguous
mapping survives past the innermost agreeing axis, and the pass pushes that
run — as little as two int32 — once per iteration of everything outside it.
Each push costs a flat ~0.5 ms at 2048 DPUs whatever it carries, so the
measured bandwidth splits into two regimes with nothing in between:

| trace | operand | MiB | pushes | scatter ms | MiB/ms |
| --- | --- | --- | --- | --- | --- |
| `mmtv_256_512_512.reproduced` | A | 256 | 16384 | 9359.4 | 0.03 |
| `mmtv_256_512_512.published` | A | 256 | 1 | 20.7 | 12.36 |

Both move the same 256 MiB to the same 2048 DPUs with the same 131072 bytes
each, and their kernels differ only in `threadIdx.x` (16 against 4).

Read the `lifted` column before comparing anything against ATiM's reported
numbers. An operand in `pragma_explicit_h2d` is lifted into its own
`copy_<symbol>` function and charged as one-time weight residency, so its
cost appears in *none* of the reported H2D/Kernel/D2H — the row above hides
9.4 seconds behind a 6.9 ms total. A transcription measured against those
numbers has to exclude the same transfer, or it loses on bookkeeping rather
than on generated code.

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
