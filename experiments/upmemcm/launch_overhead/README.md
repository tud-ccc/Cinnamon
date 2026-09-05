# Launch overhead

What `dpu_launch` costs when the kernel costs nothing.

`launch_dpu.c` is an empty program: every tasklet boots and stops. The same
binary runs on every set size, so nothing about the program varies across the
sweep and the whole measurement is the part of a launch that the cost model
does not charge — it prices instructions and DMAs, and this program has
neither. The timed call is exactly the one the generated host code makes
(`upmemrt_dpu_launch` in `runtime/Upmem/upmem_rt.c`), so the numbers are
comparable to the runtime's own `launch.csv` rows.

```
doit          # build, sweep, fit
doit bench    # just the sweep (needs the DPUs free)
```

Allocation and load are timed too, one row each per set size. They are not
part of a launch; they are here because the sweep has them anyway and their
size is worth knowing (allocation is *milliseconds*).

## What it found

50 launches per size, 8 tasklets, median (2025-08-16, upmem-2025.1.0):

| DPUs | launch (ms) | | DPUs | launch (ms) |
|-----:|------------:|-|-----:|------------:|
| 1 | 0.0495 | | 128 | 0.0581 |
| 2 | 0.0491 | | 256 | 0.0778 |
| 4 | 0.0496 | | 512 | 0.0900 |
| 8 | 0.0476 | | 1024 | 0.1275 |
| 16 | 0.0475 | | 2048 | 0.2986 |
| 32 | 0.0475 | | | |
| 64 | 0.0473 | | | |

Flat at ~0.047 ms up to 64 DPUs — one rank — and then growing with the set,
to 6x that at the full machine. Run-to-run spread is tiny (p05..p95 within
~3% of the median at every size), so the shape is not noise.

Fitted forms, least squares on the medians:

| form | RMSE (ms) | max err | coefficients |
|---|---:|---:|---|
| `a + b·log2(n) + c·n` | 0.0106 | 0.0296 | 0.0542, -0.00275, 0.000127 |
| `a + b·ranks(n)` | 0.0114 | 0.0327 | 0.0389, 0.00758 |
| `a + b·n` | 0.0123 | 0.0337 | 0.0432, 0.000115 |
| `a + b·log2(n)` | 0.0507 | 0.1409 | 0.00729, 0.0137 |
| `a` (constant) | 0.0693 | 0.2161 | 0.0825 |

A constant is out: it misses by 0.22 ms at the full machine, which is most of
the launch there. Everything with a term linear in the set size fits about
equally well, and the sweep cannot separate per-DPU from per-rank — the
machine's ranks are all full, so `ranks(n)` and `n` differ only by the factor
64. Per-rank is the form to prefer on physical grounds (~7.6 us per rank on
top of a ~39 us floor), but this data does not prove it over per-DPU.

`UpmemPythonSimulator.cpp` carries two of these shapes commented out beside
its zeroed `launchOverhead`. Both have the right form and a coefficient about
3x too large: `0.0254524·n/64` against 0.00758 per rank here, and
`0.041958·log2(n)` against 0.0137.

Allocation and load, for scale: `dpu_alloc` takes 16–123 ms (it grows with
the set, and the first allocation of a process pays extra), `dpu_load`
0.30–0.78 ms. Both are per-set, not per-launch.

## What it does not explain

The gemv_microbenchmark points measure a launch ~0.9 ms above the model's
kernel estimate at **4 DPUs** (`plots/crosscheck_functional.png`). An empty
launch at 4 DPUs costs 0.05 ms, so launch overhead accounts for about a
twentieth of that gap; the rest is something else.

At **2048 DPUs** it is the other way round: measured launches there are
0.42–0.49 ms against 0.013–0.07 ms of modelled kernel, and 0.30 ms of the
difference is the overhead measured here.
