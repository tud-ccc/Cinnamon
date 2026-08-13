"""B6: fold whatever data/ holds into results/*.csv, tolerant of holes.

One function per results table. Each globs the stacks that feed it, joins
with the offline interchange CSVs (plan §6.1) when they exist, and reports
what is MISSING instead of failing -- this is the layer that makes
"plotting with partial results" true: the plot/table scripts read only
results/*.csv and never touch data/.

Schemas (the single source of truth for the plot scripts):

- e1.csv       fn_name, system, config_label, total_ms, notes
               system in {sample, topk, search, atim_transcribed,
               atim_offline}; every *measured* config row is kept (the
               percentile in tab:sufficiency needs the whole sample
               distribution, not just its best).
- rq1.csv      the §6.1 interchange schema: benchmark, fn_name, system,
               config_label, total_ms, scatter_ms, kernel_ms, gather_ms,
               load_ms, tuning_wallclock_s, notes. Offline rows pass
               through; ours/cinm1 rows are computed here.
- rq2.csv      benchmark, fn_name, system, seed, n_candidates,
               search_wallclock_s, space_build_s, notes.
- rq3.csv      benchmark + fidelity.fidelity_frame columns (fn_name, label,
               term, predicted_ms, measured_ms, share_of_total).
- a1.csv       benchmark, fn_name, space, best_measured_ms, n_measured --
               n_measured = 0 rows are the paper's "the restricted space
               went empty" cells, printed as such, never dropped.
- rq4.csv      program, fn_name, variant, arm, config_label + one column
               per breakdown bucket (kernel a.k.a. launch, scatter*,
               gather, copy, load, unaccounted) + total_ms. Buckets use
               measurements.net_breakdown_ms(count_load=True): RQ4 is the
               undiscounted view by definition.
"""

from __future__ import annotations

import pathlib

import pandas as pd

from cinm_experiments import fidelity, measurements
from cinm_experiments.aggregate import iter_config_dirs

# §6.1: the offline interchange columns, shared by rq1.csv.
INTERCHANGE_COLUMNS = [
    "benchmark",
    "fn_name",
    "system",
    "config_label",
    "total_ms",
    "scatter_ms",
    "kernel_ms",
    "gather_ms",
    "load_ms",
    "tuning_wallclock_s",
    "notes",
]


def _write(frame: pd.DataFrame, out_csv: pathlib.Path, missing: list[str]) -> bool:
    """Write the assembled frame (when it has rows) and print the MISSING
    report either way. Never fails: an all-missing assembly is a state the
    pipeline passes through, not an error."""
    name = out_csv.name
    for note in missing:
        print(f"{name}: MISSING {note}")
    if frame.empty:
        print(f"{name}: no rows assembled yet")
        if out_csv.exists():
            # A leftover from an earlier assembly would feed the plots
            # stale rows that the current data/ no longer supports.
            out_csv.unlink()
            print(f"{name}: removed stale previous assembly")
        return True
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_csv, index=False)
    print(f"{name}: {len(frame)} rows")
    return True


def measured_rows(
    compile_root: pathlib.Path, run_root: pathlib.Path, system: str
) -> pd.DataFrame:
    """One §6.1-shaped row per benched config under a B2 stack (net total +
    component means, label from the config dir). Empty frame when the stack
    has not run."""
    rows = []
    for fn_name, config_dir in iter_config_dirs(pathlib.Path(run_root)):
        output = config_dir / "output"
        total = measurements.net_time_ms(output)
        if total is None:
            continue
        rows.append(
            {
                "fn_name": fn_name,
                "system": system,
                "config_label": config_dir.name,
                "total_ms": total,
                "scatter_ms": measurements.scatter_time_ms(output),
                "kernel_ms": measurements.launch_time_ms(output),
                "gather_ms": measurements.gather_time_ms(output),
                "load_ms": measurements.load_time_ms(output),
            }
        )
    return pd.DataFrame(rows)


def _offline_rows(offline_csv: pathlib.Path) -> pd.DataFrame | None:
    if not offline_csv.exists():
        return None
    return pd.read_csv(offline_csv)


def assemble_e1(
    stacks: dict[str, dict[str, tuple[pathlib.Path, pathlib.Path]]],
    offline_atim: pathlib.Path,
    out_csv: pathlib.Path,
) -> bool:
    """stacks: benchmark -> {system: (compile_root, run_root)} over the
    sample, topk, search and atim_transcribed stacks. Offline ATiM rows
    (already benchmark-tagged, §6.1 schema) are folded in under
    system=atim_offline."""
    frames, missing = [], []
    for bench, systems in sorted(stacks.items()):
        for system, (compile_root, run_root) in sorted(systems.items()):
            frame = measured_rows(compile_root, run_root, system)
            if frame.empty:
                missing.append(f"{system} stack of {bench} ({run_root})")
            else:
                frames.append(frame.assign(benchmark=bench))
    offline = _offline_rows(offline_atim)
    if offline is None:
        missing.append(f"offline ATiM rows ({offline_atim})")
    else:
        frames.append(
            offline.assign(system="atim_offline")[
                ["benchmark", "fn_name", "system", "config_label", "total_ms"]
            ]
        )
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return _write(frame, out_csv, missing)


def assemble_rq3(
    sample_stacks: dict[str, tuple[pathlib.Path, pathlib.Path]],
    out_csv: pathlib.Path,
) -> bool:
    """sample_stacks: benchmark -> (compile_root, run_root) of its B1/B2
    sample stack -- the shared uniform sample is RQ3's only ground truth."""
    frames, missing = [], []
    for bench, (compile_root, run_root) in sample_stacks.items():
        frame = fidelity.fidelity_frame(compile_root, run_root)
        if frame.empty:
            missing.append(f"benched sample of {bench} ({run_root})")
        else:
            frames.append(frame.assign(benchmark=bench))
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return _write(frame, out_csv, missing)


def assemble_rq2(
    search_roots: dict[str, pathlib.Path],
    space_jsons: dict[tuple[str, str], pathlib.Path],
    out_csv: pathlib.Path,
) -> bool:
    """Ours only; the ATiM walltime column is offline and joined by
    table_walltime.py. search_roots: benchmark -> B4 dump root (holding
    infer_{fn}/seed_{k}/timings.csv); space_jsons: (benchmark, fn) ->
    space.json (its space_build_seconds is RQ2's setup-time row, available
    since B0 -- so rq2.csv has rows before any search has run)."""
    rows, missing = [], []
    for (bench, fn_name), sj in sorted(space_jsons.items()):
        if not sj.exists():
            missing.append(f"space.json of {bench}:{fn_name}")
            continue
        import json

        build_s = json.loads(sj.read_text()).get("space_build_seconds")
        rows.append(
            {
                "benchmark": bench,
                "fn_name": fn_name,
                "system": "ours",
                "seed": None,
                "n_candidates": None,
                "search_wallclock_s": None,
                "space_build_s": build_s,
            }
        )
    for bench, root in sorted(search_roots.items()):
        any_seed = False
        for timings in sorted(pathlib.Path(root).glob("infer_*/seed_*/timings.csv")):
            df = pd.read_csv(timings)
            any_seed = True
            rows.append(
                {
                    "benchmark": bench,
                    "fn_name": timings.parent.parent.name.removeprefix("infer_"),
                    "system": "ours",
                    "seed": int(timings.parent.name.removeprefix("seed_")),
                    # iter is the observation count, elapsed_ms cumulative
                    # wall time -- the last row is the whole search.
                    "n_candidates": int(df["iter"].max()),
                    "search_wallclock_s": float(df["elapsed_ms"].max()) / 1000.0,
                    "space_build_s": None,
                }
            )
        if not any_seed:
            missing.append(f"search timings of {bench} ({root})")
    return _write(pd.DataFrame(rows), out_csv, missing)


def assemble_rq1(
    ours_stacks: dict[str, dict[str, tuple[pathlib.Path, pathlib.Path]]],
    offline_csvs: dict[str, pathlib.Path],
    out_csv: pathlib.Path,
) -> bool:
    """ours_stacks: benchmark -> {system: (compile_root, run_root)} for the
    measured arms (ours = search picks, falling back to whatever measured
    stacks exist; cinm1 = the (D,T) sweep). offline_csvs: system ->
    interchange CSV (prim, atim, cpu). Best-config selection is the table
    scripts' job; this keeps every measured row."""
    frames, missing = [], []
    for bench, stacks in sorted(ours_stacks.items()):
        for system, (compile_root, run_root) in sorted(stacks.items()):
            frame = measured_rows(compile_root, run_root, system)
            if frame.empty:
                missing.append(f"{system} rows of {bench} ({run_root})")
            else:
                frames.append(frame.assign(benchmark=bench))
    for system, path in sorted(offline_csvs.items()):
        offline = _offline_rows(path)
        if offline is None:
            missing.append(f"offline {system} rows ({path})")
        else:
            frames.append(offline.assign(system=system))
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not frame.empty:
        frame = frame.reindex(columns=INTERCHANGE_COLUMNS)
    return _write(frame, out_csv, missing)


def assemble_a1(
    ablate_stacks: dict[tuple[str, str], tuple[pathlib.Path, pathlib.Path]],
    out_csv: pathlib.Path,
) -> bool:
    """ablate_stacks: (benchmark, space) -> (compile_root, run_root), where
    space is one of default/no_mram/no_scatter/neither. Emits best measured
    per (benchmark, fn, space); a space whose stack ran but measured nothing
    is the infeasible cell and is kept with n_measured=0."""
    rows, missing = [], []
    for (bench, space), (compile_root, run_root) in sorted(ablate_stacks.items()):
        run_root = pathlib.Path(run_root)
        if not run_root.exists():
            missing.append(f"{space} search of {bench} ({run_root})")
            continue
        frame = measured_rows(compile_root, run_root, space)
        if frame.empty:
            # The stack ran and nothing measured: that IS the result --
            # "without this capability the space went empty" (paper A1).
            # But only per-fn directories prove the stack ran; a bare (or
            # empty) root is a stack that never started, i.e. a hole.
            fns = [d.name for d in run_root.iterdir() if d.is_dir()]
            if not fns:
                missing.append(f"{space} search of {bench} ({run_root})")
                continue
            for fn_name in fns:
                rows.append(
                    {
                        "benchmark": bench,
                        "fn_name": fn_name,
                        "space": space,
                        "best_measured_ms": None,
                        "n_measured": 0,
                    }
                )
            continue
        for fn_name, sub in frame.groupby("fn_name"):
            rows.append(
                {
                    "benchmark": bench,
                    "fn_name": fn_name,
                    "space": space,
                    "best_measured_ms": float(sub["total_ms"].min()),
                    "n_measured": len(sub),
                }
            )
    return _write(pd.DataFrame(rows), out_csv, missing)


def assemble_rq4(
    arm_stacks: dict[tuple[str, str], tuple[pathlib.Path, pathlib.Path]],
    out_csv: pathlib.Path,
) -> bool:
    """arm_stacks: (program, arm) -> (compile_root, run_root); arm is
    peroper or wholeprog. One row per benched config with the undiscounted
    breakdown (count_load=True): RQ4 prices exactly what per-operator
    isolation hides, so nothing is amortized away here."""
    rows, missing = [], []
    for (program, arm), (compile_root, run_root) in sorted(arm_stacks.items()):
        found = False
        for fn_name, config_dir in iter_config_dirs(pathlib.Path(run_root)):
            output = config_dir / "output"
            breakdown = measurements.net_breakdown_ms(output, count_load=True)
            if breakdown is None:
                continue
            found = True
            rows.append(
                {
                    "program": program,
                    "fn_name": fn_name,
                    "arm": arm,
                    "config_label": config_dir.name,
                    **{f"{k}_ms": v for k, v in breakdown.items()},
                    "total_ms": sum(breakdown.values()),
                }
            )
        if not found:
            missing.append(f"{arm} arm of {program} ({run_root})")
    return _write(pd.DataFrame(rows), out_csv, missing)
