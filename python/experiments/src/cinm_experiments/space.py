"""space.json utilities: the design space as cinm-opt dumps it.

The dump is produced by --upmem-infer-accelerator (dump-space-only=true for
a dump without a run; every search/sample mode also writes one next to its
pool.csv). This module is the Python-side reader: the evaluation pipeline
uses it for tab:sufficiency's size columns and for the A2 Cartesian
sampler, and the ATiM-transcription helper uses the permutation tables
workflow).
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import random


@dataclasses.dataclass(frozen=True)
class Param:
    """One search parameter, as space.json describes it. `dims` are the
    per-dimension names pool.csv columns and eval-solution use (equal to
    [name] for arity-1 params); permutation params additionally carry
    `items` (what is ordered) and `orderings` (every ordering with its
    exact eval-solution assignment)."""

    name: str
    kind: str  # "integer" | "permutation"
    arity: int
    cardinality: int
    num_values: float
    dims: list[str]
    doc: str = ""
    # domain: exactly one of the two forms cinm-opt dumps
    lo: int | None = None
    hi: int | None = None
    step: int | None = None
    values: list[int] | None = None
    # permutation only
    items: list[str] | None = None
    orderings: list[dict] | None = None
    encoding: str = ""

    def domain_values(self) -> list[int]:
        """Every value one dimension of this parameter can take."""
        if self.values is not None:
            return list(self.values)
        assert self.lo is not None and self.hi is not None
        return list(range(self.lo, self.hi + 1, self.step or 1))


@dataclasses.dataclass(frozen=True)
class Space:
    cartesian_size: float
    feasible_size: int
    params: list[Param]
    raw: dict  # the full parsed JSON, for whatever this class doesn't model

    @property
    def dim_names(self) -> list[str]:
        """All dimension names in declaration order -- the exact set an
        eval-solution assignment must cover."""
        return [d for p in self.params for d in p.dims]

    def param(self, name: str) -> Param:
        for p in self.params:
            if p.name == name:
                return p
        raise KeyError(
            f"no parameter named {name!r}; space has: {[p.name for p in self.params]}"
        )

    def sample_cartesian(self, n: int, *, seed: int) -> list[dict]:
        """n assignments drawn uniformly from the Cartesian product of the
        declared domains -- NOT from the feasible set. This is the A2
        rejected-region sampler's raw material: most draws violate a
        constraint, which is the point (the caller drops the feasible ones
        and attempts to lower the rest). Permutation dims are drawn as
        actual orderings (a uniform draw over the n^n place-encodings would
        mostly not be orderings at all, and 'rejected for not being a
        permutation' measures nothing about our constraint system)."""
        rng = random.Random(seed)
        out = []
        for _ in range(n):
            assignment: dict[str, int] = {}
            for p in self.params:
                if p.kind == "permutation":
                    places = list(range(1, p.arity + 1))
                    rng.shuffle(places)
                    assignment.update(zip(p.dims, places))
                else:
                    assignment[p.dims[0]] = rng.choice(p.domain_values())
            out.append(assignment)
        return out


def load(space_json: pathlib.Path) -> Space:
    raw = json.loads(pathlib.Path(space_json).read_text())
    params = []
    for p in raw["params"]:
        params.append(
            Param(
                name=p["name"],
                kind=p["kind"],
                arity=int(p["arity"]),
                cardinality=int(p["cardinality"]),
                num_values=float(p["num_values"]),
                dims=list(p.get("dims", [p["name"]])),
                doc=p.get("doc", ""),
                lo=p.get("lo"),
                hi=p.get("hi"),
                step=p.get("step"),
                values=p.get("values"),
                items=p.get("items"),
                orderings=p.get("orderings"),
                encoding=p.get("encoding", ""),
            )
        )
    return Space(
        cartesian_size=float(raw["cartesian_size"]),
        feasible_size=int(raw["feasible_size"]),
        params=params,
        raw=raw,
    )
