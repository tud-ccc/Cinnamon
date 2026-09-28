"""Reader for the graph allocator's report.

The pass writes one JSON report per graph -- `<dump-dir>/<graph>/allocation.json`,
or `<allocation-report>/<graph>.json` -- and this is the Python statement of
what it holds. Its top level:

    host, device, options   what the run priced against and ran under
    nodes                   every block of the graph: its class, location,
                            loops and predecessors
    classes                 per class of identical blocks: its operator, its
                            fate, every point of its resource menu, the
                            groups the allocation gave it, and (under
                            dump-dir) `reference`: the path, relative to the
                            report, of its one-block module and the name of
                            its function, which a point's `config` compiles
                            from on its own (eval-solution)
    allocation              what the solve achieved; null when it did not
                            get that far (a dry run, everything on the host)

A class's `points` start with the host (resource 0, priced by the host
roofline) and then list every value of its menu, each with its device
roofline, the screen's verdict, whether it was selected for a search, every
search that ran at it, and whether it reached the profile the allocation
solved over (`in_profile`; `dominated_by` says why not, when a smaller point
was no worse).

`Graph` gives the report as data frames, in the columns the per-file CSV
dumps had before it, so that what draws them does not need to know the JSON;
`points` is the one frame those dumps had no equivalent of.

Kept here rather than in whatever draws them because the format is the
compiler's, not any one figure's: a second consumer -- another plot, an
assembler -- needs the same answers.
"""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib

import pandas as pd

REPORT = "allocation.json"


@dataclasses.dataclass
class Graph:
    """One graph's report, and the frames read off it."""

    name: str
    report: dict

    @property
    def classes(self) -> list[dict]:
        return self.report["classes"]

    @property
    def points(self) -> pd.DataFrame:
        """One row per (class, point): everything that happened at each
        value of each class's menu, the host first."""
        rows = []
        for c in self.classes:
            for p in c["points"]:
                roof = p.get("roofline") or {}
                rows.append(
                    {
                        "graph": self.name,
                        "class": c["class"],
                        "fate": c["fate"],
                        "resource": p["resource"],
                        "where": p["where"],
                        "priced_by": p["priced_by"],
                        "cost_ms": p.get("cost_ms"),
                        "screen_cost_ms": p.get("screen_cost_ms"),
                        "roofline_ms": roof.get("ms"),
                        "transfer_ms": roof.get("transfer_ms"),
                        "device_ops_per_s": roof.get("ops_per_s"),
                        "transfer_ms_streamed": roof.get("transfer_ms_streamed"),
                        "screen": p.get("screen"),
                        "selected": p.get("selected", False),
                        "in_profile": p["in_profile"],
                        "dominated_by": p.get("dominated_by"),
                        "transfer_share": p.get("transfer_share"),
                    }
                )
        return pd.DataFrame(rows)

    @property
    def profiles(self) -> pd.DataFrame:
        """What the allocator solved over: one row per (class, point in its
        profile), and one measurement-less row per class that never entered
        the solve, so the frame names every class of the graph."""
        levels = [lvl["name"] for lvl in self.report["device"]["levels"]]
        rows = []
        for c in self.classes:
            base = {
                "class": c["class"],
                "debug_tag": c["debug_tag"],
                "location": c["loc"],
                "multiplicity": c["members"],
            }
            kept = [p for p in c["points"] if p["in_profile"]]
            if not kept:
                rows.append(base)
                continue
            for p in kept:
                row = dict(base)
                row["resource"] = p["resource"]
                row["cost_ms"] = p["cost_ms"]
                row["transfer_share"] = p.get("transfer_share")
                row["weight_scatter_ms"] = p.get("weight_scatter_ms", 0.0)
                config = p.get("config") or {}
                row["config"] = ";".join(f"{k}={v}" for k, v in sorted(config.items()))
                residency = p.get("residency") or {}
                for level in levels + [lvl for lvl in residency if lvl not in levels]:
                    entry = residency.get(level)
                    row[f"static_{level}"] = entry["static_bytes"] if entry else None
                    row[f"dyn_{level}"] = entry["dyn_bytes"] if entry else None
                rows.append(row)
        return pd.DataFrame(rows)

    @property
    def groups(self) -> pd.DataFrame | None:
        """The device sets the allocation carved out, one row each; None when
        it did not get that far. `resource` is what a set reserves (0 for a
        timeshared or host group), `point_resource` the profile point it
        runs."""
        if self.report["allocation"] is None:
            return None
        rows = [
            {
                "graph": self.name,
                "class": c["class"],
                "group": g["group"],
                "size": g["size"],
                "resource": g["resource"],
                "point_resource": g["point_resource"],
                "cost_ms": g["cost_ms"],
                "load_ms": g["load_ms"],
                "timeshared": int(g["timeshared"]),
                "on_host": int(g["on_host"]),
            }
            for c in self.classes
            for g in c["groups"] or []
        ]
        return pd.DataFrame(rows)

    @property
    def alloc(self) -> pd.Series | None:
        """The summary of what the allocation decided, or None when it did
        not get that far. The host counts are the classes and blocks that
        never entered the solve; a group the solve itself placed on the host
        is counted in `n_groups_host`."""
        a = self.report["allocation"]
        if a is None:
            return None
        solved = [c for c in self.classes if c["groups"] is not None]
        groups = [g for c in solved for g in c["groups"]]
        n_blocks = len(self.report["nodes"])
        n_device = sum(c["members"] for c in solved)
        return pd.Series(
            {
                "graph": self.name,
                "platform": self.report["platform"],
                "objective": self.report["options"]["objective"],
                "objective_ms": a["objective_ms"],
                "throughput_ms": a["throughput_ms"],
                "latency_ms": math.nan if a["latency_ms"] is None else a["latency_ms"],
                "n_blocks": n_blocks,
                "n_blocks_device": n_device,
                "n_blocks_host": n_blocks - n_device,
                "n_classes": len(self.classes),
                "n_classes_device": len(solved),
                "n_classes_host": len(self.classes) - len(solved),
                "n_groups": len(groups),
                "n_groups_pinned": sum(1 for g in groups if g["resource"] > 0),
                "n_groups_timeshared": sum(1 for g in groups if g["timeshared"]),
                "n_groups_host": sum(1 for g in groups if g["on_host"]),
                "resource_used": a["resource_used"],
                "resource_budget": a["resource_budget"],
            }
        )

    @property
    def seeds(self) -> pd.DataFrame | None:
        """Every search that found something, one row per (class, menu
        point, repeat); None when every point was searched once, since there
        is then no spread to read."""
        if self.report["options"]["profile_seeds"] <= 1:
            return None
        rows = [
            {
                "graph": self.name,
                "class": c["class"],
                "resource": p["resource"],
                "seed": s["repeat"],
                "cost_ms": s["cost_ms"],
            }
            for c in self.classes
            for p in c["points"]
            for s in p.get("searches", [])
            if s["cost_ms"] is not None
        ]
        return pd.DataFrame(rows)

    @property
    def menu_screen(self) -> pd.DataFrame:
        """What the menu screen made of every class: one row per (class,
        candidate resource value), with the host time it compared against
        (already slowed by the host's achieved fraction), the device
        roofline and its two terms, and the verdict. A class the screen
        could not price has its values kept with zero rooflines, as the
        screen keeps them. The host model priced against is in every row
        (host_ops_per_s, host_dram_bytes_per_s, host_fraction), so that a
        figure of the rooflines draws the ones the screen used."""
        model = self.report["host"]
        rows = []
        for c in self.classes:
            f = c["footprint"]
            menu = [p for p in c["points"] if p["where"] == "device"]
            host = next((p for p in c["points"] if p["where"] == "host"), None)
            host_ms = (host or {}).get("screen_cost_ms") or 0.0
            survivors = sum(1 for p in menu if p["screen"] != "dropped")
            for p in menu:
                roof = p.get("roofline") or {}
                rows.append(
                    {
                        "graph": self.name,
                        "class": c["class"],
                        "blocks": c["members"],
                        "loc": c["loc"],
                        "work_ops": f.get("work_ops", 0.0),
                        "static_bytes": f.get("static_bytes", 0.0),
                        # What the device holds, where static_bytes is what
                        # one execution reads (a layer of a stacked weight).
                        "static_resident_bytes": f.get(
                            "static_resident_bytes", f.get("static_bytes", 0.0)
                        ),
                        "dynamic_bytes": f.get("dynamic_bytes", 0.0),
                        "dynamic_out_bytes": f.get("dynamic_out_bytes", 0.0),
                        "host_ms": host_ms,
                        "host_ops_per_s": model["ops_per_s"],
                        "host_dram_bytes_per_s": model["dram_bytes_per_s"],
                        "host_fraction": model["achieved_fraction"],
                        "resource": p["resource"],
                        "device_ms": roof.get("ms", 0.0),
                        "transfer_ms": roof.get("transfer_ms", 0.0),
                        "device_ops_per_s": roof.get("ops_per_s", 0.0),
                        "transfer_ms_streamed": roof.get("transfer_ms_streamed", 0.0),
                        "kept": int(p["screen"] != "dropped"),
                        "candidates": len(menu),
                        "kept_of_candidates": survivors,
                        "profiled": int(p["selected"]),
                    }
                )
        return pd.DataFrame(rows)

    def spread(self, class_ix: int) -> pd.DataFrame | None:
        """min/max cost per menu point over the repeated searches of one
        class, or None when the profile was measured once (nothing to band)."""
        seeds = self.seeds
        if seeds is None or seeds.empty:
            return None
        cls = seeds[seeds["class"] == class_ix]
        if cls.empty or cls["seed"].nunique() < 2:
            return None
        return cls.groupby("resource")["cost_ms"].agg(["min", "max"])


def load(path: pathlib.Path) -> Graph:
    """The report at `path`."""
    report = json.loads(path.read_text())
    return Graph(name=report["graph"], report=report)


def _is_report(path: pathlib.Path) -> bool:
    try:
        head = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return isinstance(head, dict) and "schema" in head and "classes" in head


def reports(paths: list[pathlib.Path]) -> list[pathlib.Path]:
    """Every report under `paths`. A path may be a report itself, a dump-dir
    (whose graphs each hold an allocation.json), an allocation-report
    directory (whose reports are named after their graphs), or any directory
    above those."""
    found: list[pathlib.Path] = []
    for path in paths:
        if path.is_file():
            found.append(path)
            continue
        dumped = sorted(path.rglob(REPORT))
        found.extend(dumped)
        if not dumped:
            found.extend(p for p in sorted(path.rglob("*.json")) if _is_report(p))
    return found


def collect(paths: list[pathlib.Path]) -> list[Graph]:
    """Every graph reported under `paths` (see `reports`) that has anything
    to draw: at least one class with a profile."""
    graphs = [load(p) for p in reports(paths)]
    return [
        g
        for g in graphs
        if any(p["in_profile"] for c in g.classes for p in c["points"])
    ]
