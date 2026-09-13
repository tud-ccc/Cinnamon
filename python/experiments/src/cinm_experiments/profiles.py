"""Reader for the graph allocator's dump.

A solved graph writes four files side by side, and this is the only statement
of what they are and how they relate:

    profiles.csv        per class, the cost of its best configuration at each
                        point of the shared-resource menu -- the input the
                        allocator solves over
    allocation.csv      one row: the summary of what it decided
    groups.csv          the device sets it produced
    profile_seeds.csv   present when profile-seeds > 1: the same profiles
                        measured by independent searches, for a noise band

Only profiles.csv is guaranteed. A solve that found nothing feasible stops
after it, so the other three are read as optional rather than missing.

Kept here rather than in whatever draws them because the format is the
compiler's, not any one figure's: a second consumer -- another plot, a test
over a dumped header -- needs the same answers.
"""

from __future__ import annotations

import dataclasses
import pathlib

import pandas as pd

PROFILES = "profiles.csv"
ALLOCATION = "allocation.csv"
GROUPS = "groups.csv"
SEEDS = "profile_seeds.csv"


@dataclasses.dataclass
class Graph:
    """One solved graph: the profiles the allocator was given, and -- when
    the solve got that far -- the summary and the device sets it produced."""

    name: str
    profiles: pd.DataFrame
    alloc: pd.Series | None
    groups: pd.DataFrame | None
    seeds: pd.DataFrame | None

    def spread(self, class_ix: int) -> pd.DataFrame | None:
        """min/max cost per menu point over the repeated searches of one
        class, or None when the profile was measured once (nothing to band)."""
        if self.seeds is None:
            return None
        cls = self.seeds[self.seeds["class"] == class_ix]
        if cls.empty or cls["seed"].nunique() < 2:
            return None
        return cls.groupby("resource")["cost_ms"].agg(["min", "max"])


def _read(path: pathlib.Path) -> pd.DataFrame | None:
    """A sibling dump, or None when this graph did not get that far."""
    return pd.read_csv(path) if path.exists() else None


def load(path: pathlib.Path) -> Graph | None:
    """The graph dumped into `path`, or None when it holds no profiles.

    The graph's name is the directory the pass dumped it into, which is what
    the pass named the graph.
    """
    profiles = pd.read_csv(path / PROFILES)
    if profiles.empty:
        return None
    alloc = _read(path / ALLOCATION)
    return Graph(
        name=path.name,
        profiles=profiles,
        alloc=None if alloc is None or alloc.empty else alloc.iloc[0],
        groups=_read(path / GROUPS),
        seeds=_read(path / SEEDS),
    )


def collect(paths: list[pathlib.Path]) -> list[Graph]:
    """Every graph under `paths`. A path may be a profiles.csv itself, the
    directory holding one, or any directory above such directories."""
    dirs: list[pathlib.Path] = []
    for path in paths:
        if path.is_file():
            dirs.append(path.parent)
        elif (path / PROFILES).exists():
            dirs.append(path)
        else:
            dirs.extend(sorted(p.parent for p in path.rglob(PROFILES)))
    return [g for g in (load(d) for d in dirs) if g is not None]
