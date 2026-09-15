"""Where this package finds the things outside itself.

Two kinds of path, found two ways.

*Package resources* -- the makefile compile_run.py shells out to -- are
addressed relative to this file. They ship inside the distribution and are
correct however it was installed.

*Everything a Cinnamon build produced* -- the tools, the benchmark suites, the
UPMEM runtime, the reference model -- comes from `cinnamon.paths`, which knows
where a build was installed. Nothing here joins a path onto a checkout root,
which is what used to tie a campaign to one clone: the pipelines ask by name
and get whatever build the environment holds.
"""

from __future__ import annotations

import pathlib
import sys

# Package data, not a location in a repository: compile_run.py runs `make -C`
# here, and the makefile is shipped with the package (see pyproject.toml).
COMPILE_MAKEFILE_DIR = pathlib.Path(__file__).resolve().parent


def _cinnamon():
    """cinnamon.paths, imported on use rather than at module load.

    Reading a campaign's results needs no compiler, and the environments that
    do that have none installed -- importing this module must not be what
    tells them so. A caller that actually needs a build gets cinnamon's own
    error, which says where it looked.
    """
    from cinnamon import paths

    return paths


def cinm_opt() -> pathlib.Path:
    """The compiler every pipeline drives."""
    return _cinnamon().tool("cinm-opt")


def bin_dir() -> pathlib.Path:
    """Where cinm-opt and the tools beside it are: cinm-translate and
    cinm-compile-dpu, plus the opt and llc of the LLVM this was built
    against. The makefile wants the directory rather than the tools, so it
    can name them itself."""
    return _cinnamon().bin_dir()


def benchmarks_dir() -> pathlib.Path:
    """The benchmark suites, one directory per family: prim/ for the
    PrIM-derived single operators, multiop/ for the multi-operator graphs,
    cinm1/ for the CINM 1.0 suite. Each holds a <name>.mlir beside the
    <name>.cpp driver that runs it."""
    return _cinnamon().benchmarks_dir()


def bench_source(bench: str) -> pathlib.Path:
    """The source module of a benchmark, by the name the pipelines know it by.

    That name is not the file name: a prim is `prim_gemv` throughout -- task
    ids, data/ directories, the doit database -- while its source is
    prim/gemv.mlir. Keeping the two independent is what let the suite move
    without invalidating a campaign's state, so resolve through here rather
    than reconstructing either from the other.
    """
    root = benchmarks_dir()
    if bench.startswith("prim_"):
        return root / "prim" / f"{bench.removeprefix('prim_')}.mlir"
    return root / "multiop" / f"{bench}.mlir"


def bench_driver(prim: str) -> pathlib.Path:
    """The C++ driver that runs a benchmark, by its `compile_run.Config.prim`
    -- the bare operator name for a prim (`gemv`), the workload name for a
    multi-op graph (`2mm_seq`). The makefile resolves this itself; this is for
    callers that need to name the file."""
    root = benchmarks_dir()
    candidate = root / "prim" / f"{prim}.cpp"
    return candidate if candidate.exists() else root / "multiop" / f"{prim}.cpp"


def reference_model_dir() -> pathlib.Path:
    """The analytical cost model refmodel.py prices programs against."""
    return _cinnamon().reference_model_dir()


def python_bin() -> str:
    """The interpreter to run a helper script with: this one. The pipelines
    and the scripts they spawn belong to the same environment, which is the
    one this is running in."""
    return sys.executable
