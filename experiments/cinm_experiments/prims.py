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
import pathlib

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
    its per-function problem sizes."""

    name: str
    dimensions: dict[str, dict[str, int]]

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


PRIMS: dict[str, Prim] = {
    # Reduction of a vector of K elements.
    "red": Prim(
        name="red",
        dimensions={
            "4MB": dict(K=524288),
            "64MB": dict(K=8388608),
            "256MB": dict(K=34554432),
            "512MB": dict(K=67108864),
        },
    ),
    # Elementwise vector add of two K-element vectors (geva scales them
    # first); both top out at 256MB -- there is no 512MB kernel.
    "va": Prim(
        name="va",
        dimensions={
            "4MB": dict(K=1048576),
            "64MB": dict(K=16777216),
            "256MB": dict(K=67108864),
        },
    ),
    "geva": Prim(
        name="geva",
        dimensions={
            "4MB": dict(K=1048576),
            "64MB": dict(K=16777216),
            "256MB": dict(K=67108864),
        },
    ),
    # Matrix (MxK) times vector (K); gemv also scales the result.
    "gemv": Prim(
        name="gemv",
        dimensions={
            "4MB": dict(M=1024, K=1024),
            "64MB": dict(M=4096, K=4096),
            "256MB": dict(M=8192, K=8192),
            "512MB": dict(M=8192, K=16384),
        },
    ),
    "mtv": Prim(
        name="mtv",
        dimensions={
            "4MB": dict(M=1024, K=1024),
            "64MB": dict(M=4096, K=4096),
            "256MB": dict(M=8192, K=8192),
            "512MB": dict(M=8192, K=16384),
        },
    ),
    # Batch of B (MxK) matrices times a vector: mmtv one vector per batch
    # element (BxK), ttv the same K-element vector for all of them.
    "mmtv": Prim(
        name="mmtv",
        dimensions={
            "4MB": dict(B=32, M=64, K=512),
            "64MB": dict(B=128, M=256, K=512),
            "256MB": dict(B=256, M=512, K=512),
            "512MB": dict(B=512, M=512, K=512),
        },
    ),
    "ttv": Prim(
        name="ttv",
        dimensions={
            "4MB": dict(B=32, M=64, K=512),
            "64MB": dict(B=128, M=256, K=512),
            "256MB": dict(B=256, M=512, K=512),
            "512MB": dict(B=512, M=512, K=512),
        },
    ),
}
