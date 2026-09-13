"""Where this package finds the things outside itself.

Three kinds of path live here, and they are found in three different ways,
because only one of them is the package's own.

*Package resources* (the makefile) are addressed relative to this file. They
ship inside the distribution and are correct however it was installed.

*Tools* (cinm-opt) are configuration: an environment variable, defaulting to
the build tree of the checkout this package was installed from. A caller that
built elsewhere sets the variable.

*Repository data* (the benchmark suites, the cost model's Predictor) is not
this package's property at all. It is found through CINM_MLIR_ROOT, which
falls back to walking up from this file -- valid only for an in-tree or
editable install, which is how it is used today. That fallback is why a
non-editable install into an unrelated environment must set the variable: the
package would otherwise look for `benchmarks/` inside site-packages.
"""

import os
import pathlib
import sys

# src/cinm_experiments/paths.py -> src/cinm_experiments -> src -> experiments
# -> python -> the repository root.
_IN_TREE_ROOT = pathlib.Path(__file__).resolve().parents[4]
ROOT = pathlib.Path(os.environ.get("CINM_MLIR_ROOT", _IN_TREE_ROOT))

DEFAULT_CINM_OPT = pathlib.Path(
    os.environ.get("CINM_OPT", ROOT / "build" / "bin" / "cinm-opt")
)

# Package data, not a location in the repository: compile_run.py runs `make -C`
# here, and the makefile is shipped with the package (see pyproject.toml).
COMPILE_MAKEFILE_DIR = pathlib.Path(__file__).resolve().parent

# The benchmark suites: each <name>.mlir next to the <name>.cpp driver that
# runs it, one directory per family -- prim/ for the PrIM-derived single
# operators, multiop/ for the multi-operator graphs (cinm1/ is a third, built
# by its own makefile rather than through here). Every pipeline resolves its
# sources through these rather than joining paths itself, so the suites can be
# rearranged without touching them.
BENCH_DIR = ROOT / "benchmarks"
PRIM_BENCH_DIR = BENCH_DIR / "prim"
MULTIOP_BENCH_DIR = BENCH_DIR / "multiop"


def bench_source(bench: str) -> pathlib.Path:
    """The source module of a benchmark, by the name the pipelines know it by.

    That name is not the file name: a prim is `prim_gemv` throughout -- task
    ids, data/ directories, the doit database -- while its source is
    prim/gemv.mlir. Keeping the two independent is what let the suite move
    without invalidating a campaign's state, so resolve through here rather
    than reconstructing either from the other.
    """
    if bench.startswith("prim_"):
        return PRIM_BENCH_DIR / f"{bench.removeprefix('prim_')}.mlir"
    return MULTIOP_BENCH_DIR / f"{bench}.mlir"


def bench_driver(prim: str) -> pathlib.Path:
    """The C++ driver that runs a benchmark, by its `compile_run.Config.prim`
    -- the bare operator name for a prim (`gemv`), the workload name for a
    multi-op graph (`2mm_seq`). The Makefile resolves this itself; this is for
    callers that need to name the file."""
    candidate = PRIM_BENCH_DIR / f"{prim}.cpp"
    return candidate if candidate.exists() else MULTIOP_BENCH_DIR / f"{prim}.cpp"


def python_bin() -> str:
    """The experiments venv's python if it exists, else the current interpreter."""
    venv = ROOT / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable
