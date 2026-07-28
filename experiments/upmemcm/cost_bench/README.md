This is an E2E benchmark to validate the DPU cost model: for one or more
primitives (currently `red`, `gemv`), it compiles and runs every valid config
from an existing cost-model oracle `pool.csv` (under `experiments/data/`) on
real UPMEM hardware, then plots measured vs predicted cost for calibration.

Driven by `doit` (see `dodo.py`'s module docstring for the full task
breakdown and usage). Quick start:

```
doit list           # show all tasks
doit                # split -> compile -> bench -> aggregate -> plot (default)
doit compile:red    # just compile red's configs (no hardware needed)
```

Adding a new benchmark = one more entry in `dodo.py`'s `PRIMS` dict (source
MLIR + oracle dir + a config filter to keep the hardware sweep tractable)
plus an `experiments/bench/<prim>.cpp` driver; nothing else needs to change.

`plot_cost.py`, `compare_oracles.py`, `fit_overhead_term.py`, and
`plot_transfer_cost.py` are standalone analysis/plotting scripts specific to
this experiment (also runnable directly, not just via `doit plot`).
