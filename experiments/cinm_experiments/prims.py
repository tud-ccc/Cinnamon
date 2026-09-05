"""The benchmark primitives (red, gemv, ...) every experiment pipeline
shares: for each one, the multi-function source module holding its kernels
and the problem dimensions of each of those functions.

One Prim per experiments/prim_<name>.mlir + experiments/bench/<name>.cpp
pair; one `dimensions` entry per function in that module, keyed by the
function's size suffix ("4MB", "64MB", ...) so fn_names() can reconstruct
the MLIR function names. The dimension names are the same ones the kernels'
tensor shapes use -- M rows, K the contiguous/reduced innermost extent, B
the batch -- so a config's problem size can be talked about without
re-parsing the MLIR.

Adding a benchmark = one more PRIMS entry, plus the prim_<name>.mlir and
bench/<name>.cpp it names; no pipeline code needs to change.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import pathlib
import re

from .paths import EXPERIMENTS_DIR
from .split_source import list_functions


# eq=False: `dimensions` is a dict, so the generated __eq__/__hash__ would be
# unusable (unhashable field) -- and these are module-level singletons anyway,
# for which identity is the right notion of equality. It's what lets
# _check_source() below be an ordinary functools.cache'd function.
@dataclasses.dataclass(frozen=True, eq=False)
class Prim:
    """One benchmark primitive. `name` is both the bench driver's op name
    (bench/<name>.cpp, i.e. compile_run.Config.prim) and the prim_<name>.mlir
    stem -- everything else is derived from it, so a Prim is just a name plus
    its per-function problem sizes.

    `parallel_dims` names which of `dimensions` the workgroup is spread over,
    as opposed to the ones a single worker walks by itself -- for a gemv, its
    M rows are handed out to the workgroup while every worker walks all K of
    its row's columns. That split is a property of the kernel, and it is what
    working_groups() enumerates against; see its docstring."""

    name: str
    dimensions: dict[str, dict[str, int]]
    parallel_dims: tuple[str, ...]

    @property
    def source_mlir(self) -> pathlib.Path:
        return EXPERIMENTS_DIR / f"prim_{self.name}.mlir"

    def fn_names(self) -> tuple[str, ...]:
        """Every function of this prim's source module, in `dimensions`
        order. Derived from `dimensions` rather than parsed out of the
        module so it is available before anything has read (or split, or
        searched) the source; _check_source() guards the two against drift.
        """
        _check_source(self)
        return tuple(f"{self.name}_{label}" for label in self.dimensions)

    def dims(self, fn_name: str) -> dict[str, int]:
        """The problem dimensions of one of this prim's functions, e.g.
        dims("gemv_4MB") -> {"M": 1024, "K": 1024}."""
        return self.dimensions[fn_name.removeprefix(self.name + "_")]

    def parallel_extent(self, fn_name: str) -> int:
        """How much parallel work one of this prim's functions has: the
        product of its `parallel_dims`, e.g. parallel_extent("mmtv_4MB") ->
        B*M = 2048. The number of workers a config can distribute the kernel
        over exactly -- see working_groups()."""
        dims = self.dims(fn_name)
        return math.prod(dims[d] for d in self.parallel_dims)

    def working_groups(
        self, fn_name: str, *, max_dpus: int, max_tasklets: int
    ) -> list[tuple[int, int]]:
        """Every (dpus, tasklets) working group that can run `fn_name` at all,
        in (dpus, tasklets) order: the ones whose dpus*tasklets workers divide
        the function's parallel_extent(), so the parallel work splits evenly
        between them with none left over and no worker idle.

        This is a structural property of the problem, not a prediction about
        it -- so it takes no cost model and rules nothing out on performance
        grounds, which is what makes it usable as the *full* set of working
        groups worth measuring rather than a screened-down subset. A group
        outside it cannot be compiled: CINM 1.0 refuses to infer tile sizes
        for it (TilingParameters::parallelClusterSize), and CINM 2.0's search
        space is empty for it (`prod(extent / block) == dpus * tasklets` over
        block sizes that divide their extent). The converse does not hold --
        a group in here can still turn out not to fit in WRAM, which only
        compiling it tells you."""
        extent = self.parallel_extent(fn_name)
        return [
            (dpus, tasklets)
            for dpus in range(1, max_dpus + 1)
            if extent % dpus == 0
            for tasklets in range(1, max_tasklets + 1)
            if extent % (dpus * tasklets) == 0
        ]


_PLATFORM_RE = re.compile(
    r"#upmem\.platform<[^>]*\bdpus\s*=\s*(\d+)[^>]*\btasklets\s*=\s*(\d+)"
)


@functools.cache
def platform_limits(prim: Prim) -> tuple[int, int]:
    """The (max_dpus, max_tasklets) the prim's source module declares, read
    off its `#upmem.platform<...>` attributes -- the bounds working_groups()
    enumerates within. Taken from the source rather than hardcoded next to
    it, since that attribute is what the compiler itself bounds a
    configuration by; a hardcoded pair here could drift from it silently and
    hand the pipeline working groups no config can use. Every function of a
    module must declare the same platform (one module = one machine)."""
    declared = set(_PLATFORM_RE.findall(prim.source_mlir.read_text()))
    if not declared:
        raise ValueError(f"{prim.source_mlir}: no #upmem.platform attribute found")
    if len(declared) > 1:
        raise ValueError(
            f"{prim.source_mlir} declares {len(declared)} different platforms "
            f"({sorted(declared)}); the pipeline assumes one per module"
        )
    dpus, tasklets = declared.pop()
    return int(dpus), int(tasklets)


@functools.cache
def _check_source(prim: Prim) -> None:
    """Fail loudly if `dimensions` and the source module have drifted apart
    -- a function present in only one of the two would otherwise silently
    either never get benchmarked or KeyError deep inside a doit task. Cached:
    one read of each source module per process, even though fn_names() is
    called from every task generator."""
    declared = sorted(f"{prim.name}_{label}" for label in prim.dimensions)
    in_source = sorted(list_functions(prim.source_mlir))
    if declared != in_source:
        raise ValueError(
            f"{prim.source_mlir} declares {in_source}, but PRIMS[{prim.name!r}]"
            f".dimensions describes {declared}"
        )
    for label, dims in prim.dimensions.items():
        unknown = set(prim.parallel_dims) - set(dims)
        if unknown:
            raise ValueError(
                f"PRIMS[{prim.name!r}].parallel_dims names {sorted(unknown)}, "
                f"which {prim.name}_{label} has no extent for ({sorted(dims)})"
            )


PRIMS: dict[str, Prim] = {
    # Reduction of a vector of K elements. K is the reduced dimension, but it
    # is also the one that is distributed: workers reduce disjoint chunks of
    # the vector in parallel and their partial results are combined
    # afterwards, so it counts as parallel here.
    "red": Prim(
        name="red",
        dimensions={
            "4MB": dict(K=524288),
            "64MB": dict(K=8388608),
            "256MB": dict(K=34554432),
            "512MB": dict(K=67108864),
        },
        parallel_dims=("K",),
    ),
    # Elementwise vector add of two K-element vectors (geva scales them
    # first); both top out at 256MB -- there is no 512MB kernel. Elementwise,
    # so every element is independent and K is parallel.
    "va": Prim(
        name="va",
        dimensions={
            "4MB": dict(K=1048576),
            "64MB": dict(K=16777216),
            "256MB": dict(K=67108864),
        },
        parallel_dims=("K",),
    ),
    "geva": Prim(
        name="geva",
        dimensions={
            "4MB": dict(K=1048576),
            "64MB": dict(K=16777216),
            "256MB": dict(K=67108864),
        },
        parallel_dims=("K",),
    ),
    # Matrix (MxK) times vector (K); gemv also scales the result. Rows are
    # handed out to the workgroup; K is reduced within a worker.
    "gemv": Prim(
        name="gemv",
        dimensions={
            "4MB": dict(M=1024, K=1024),
            "64MB": dict(M=4096, K=4096),
            "256MB": dict(M=8192, K=8192),
            "512MB": dict(M=8192, K=16384),
        },
        parallel_dims=("M",),
    ),
    "mtv": Prim(
        name="mtv",
        dimensions={
            "4MB": dict(M=1024, K=1024),
            "64MB": dict(M=4096, K=4096),
            "256MB": dict(M=8192, K=8192),
            "512MB": dict(M=8192, K=16384),
        },
        parallel_dims=("M",),
    ),
    # Batch of B (MxK) matrices times a vector: mmtv one vector per batch
    # element (BxK), ttv the same K-element vector for all of them. Both
    # batch and rows are parallel, K is reduced within a worker.
    "mmtv": Prim(
        name="mmtv",
        dimensions={
            "4MB": dict(B=32, M=64, K=512),
            "64MB": dict(B=128, M=256, K=512),
            "256MB": dict(B=256, M=512, K=512),
            "512MB": dict(B=512, M=512, K=512),
        },
        parallel_dims=("B", "M"),
    ),
    "ttv": Prim(
        name="ttv",
        dimensions={
            "4MB": dict(B=32, M=64, K=512),
            "64MB": dict(B=128, M=256, K=512),
            "256MB": dict(B=256, M=512, K=512),
            "512MB": dict(B=512, M=512, K=512),
        },
        parallel_dims=("B", "M"),
    ),
}
