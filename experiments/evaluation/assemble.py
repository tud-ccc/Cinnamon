"""B6: fold whatever data/ holds into results/*.csv, tolerant of holes.

One function per results table. Each globs the stacks that feed it, joins
with the offline interchange CSVs when they exist, and reports
what is MISSING instead of failing -- this is the layer that makes
"plotting with partial results" true: the plot/table scripts read only
results/*.csv and never touch data/.

Schemas (the single source of truth for the plot scripts):

- e1.csv       fn_name, system, config_label, total_ms,
               excluded_transfer_ms, excluded_transfer_bytes, notes
               system in {sample, topk, search, atim_{published,
               reproduced}, atim_{published,reproduced}_transcribed};
               every *measured* config row is kept (the percentile in
               tab:sufficiency needs the whole sample distribution, not
               just its best). The excluded columns carry each row's own
               convention, so a comparison across systems can check that
               they agree instead of assuming it.
- rq1.csv      the offline interchange schema: benchmark, fn_name, system,
               config_label, total_ms, scatter_ms, kernel_ms, gather_ms,
               load_ms, excluded_transfer_ms, excluded_transfer_bytes,
               tuning_wallclock_s, notes. Offline rows pass through;
               ours/cinm1 rows are computed here.
- rq2.csv      benchmark, fn_name, system, seed, n_candidates,
               search_wallclock_s, space_build_s, notes.
- rq3.csv      benchmark + fidelity.fidelity_frame columns (fn_name, label,
               term, predicted_ms, measured_ms, share_of_total).
- sample_census.csv
               benchmark, fn_name, n_requested, n_accepted, n_rows,
               n_timed_out, n_timed_out_in_pool, n_failed, n_over_budget,
               space_size, sampling_mode -- what the B1 draw did, which is
               what the percentile's confidence bound is computed against.
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

import json
import pathlib

import numpy as np
import pandas as pd

from cinm_experiments import fidelity, measurements
from cinm_experiments.aggregate import iter_config_dirs

# The offline interchange columns (offline/{atim,prim,cpu}.csv), shared
# by rq1.csv. One row per measured point, component columns nullable;
# nothing else in the pipeline knows how offline rows were obtained.
#
# `excluded_transfer_ms` is the operand movement a system performs but does
# not report -- ATiM's `pragma_explicit_h2d` operands, our `cinm.static`
# ones. It is not part of `total_ms` on either side, by construction: that is
# what makes it worth its own column rather than a component of one. A
# single-operator benchmark can only justify leaving it out if there is
# something to amortize it against, which is a property of the workload
# (`mtv`'s weight has it, `va`'s operands do not), so the number has to
# travel per row rather than be assumed.
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
    "excluded_transfer_ms",
    "excluded_transfer_bytes",
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
    """One interchange-shaped row per benched config under a B2 stack (net total +
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
                # The scatter net_time_ms discounts, reported rather than
                # left implicit in that default: it is the same quantity
                # ATiM's pragma_explicit_h2d operands contribute to its
                # column, and a comparison against them is only sound if
                # both sides exclude or both include. Zero is the answer
                # for a function with no `cinm.static` operand (`va`), not
                # a missing measurement.
                "excluded_transfer_ms": measurements.amortizable_time_ms(
                    output, "scatter"
                ),
                "excluded_transfer_bytes": measurements.amortizable_transfer_bytes(
                    output, "scatter"
                ),
            }
        )
    return pd.DataFrame(rows)


def _offline_rows(
    offline_csv: pathlib.Path, default_system: str
) -> pd.DataFrame | None:
    """Rows from one offline interchange CSV, each tagged with its system.

    One file may hold several: ATiM's holds the schedules published with
    their artifact and the ones we reproduced by tuning here, which were
    produced by different searches on different machines and are only
    comparable as separate systems. The file says which row is which, so
    its `system` column wins; `default_system` names the rows of a file
    that does not distinguish any (it is also what the MISSING report
    calls the file).
    """
    if not offline_csv.exists():
        return None
    frame = pd.read_csv(offline_csv)
    if "system" not in frame.columns:
        return frame.assign(system=default_system)
    return frame.assign(system=frame["system"].fillna(default_system))


def assemble_e1(
    stacks: dict[str, dict[str, tuple[pathlib.Path, pathlib.Path]]],
    offline_atim: pathlib.Path,
    out_csv: pathlib.Path,
) -> bool:
    """stacks: benchmark -> {system: (compile_root, run_root)} over the
    sample, topk, search and per-variant transcription stacks. Offline ATiM rows
    (already benchmark-tagged, interchange schema) are folded in under
    the system each names -- atim_published and atim_reproduced -- or
    under atim_offline if the file does not distinguish them."""
    frames, missing = [], []
    for bench, systems in sorted(stacks.items()):
        for system, (compile_root, run_root) in sorted(systems.items()):
            frame = measured_rows(compile_root, run_root, system)
            if frame.empty:
                missing.append(f"{system} stack of {bench} ({run_root})")
            else:
                frames.append(frame.assign(benchmark=bench))
    offline = _offline_rows(offline_atim, "atim_offline")
    if offline is None:
        missing.append(f"offline ATiM rows ({offline_atim})")
    else:
        frames.append(
            offline[
                [
                    "benchmark",
                    "fn_name",
                    "system",
                    "config_label",
                    "total_ms",
                    "excluded_transfer_ms",
                    "excluded_transfer_bytes",
                ]
            ]
        )
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return _write(frame, out_csv, missing)


def assemble_sample_census(
    stats_jsons: dict[tuple[str, str], pathlib.Path],
    pool_csvs: dict[tuple[str, str], pathlib.Path],
    out_csv: pathlib.Path,
) -> bool:
    """What the B1 draw actually did, per (benchmark, fn): how many
    configurations it requested, how many it kept, and how many of those
    carry no prediction because the simulator ran out of its budget.

    This is the input to the percentile's conservative reading. The rule of
    three assumes n independent trials each of which we can score against the
    reference; a timed-out row is a trial we drew and measured but cannot
    score from the predicted side, so the honest bound is the one that scores
    every one of them against us. That needs the count, which pool.csv only
    half-carries -- `timed_out` is recoverable from its non-finite costs, but
    `failed` and `over_budget` left no row at all.

    `n_rows` is read back from pool.csv rather than trusted from the census,
    because it is the pool that feeds B2: the two disagreeing means the draw
    and the thing downstream measured are not the same sample."""
    rows, missing = [], []
    for (bench, fn_name), stats_path in sorted(stats_jsons.items()):
        if not pathlib.Path(stats_path).exists():
            missing.append(f"sample census of {bench}:{fn_name} ({stats_path})")
            continue
        with open(stats_path) as f:
            stats = json.load(f)
        pool_csv = pathlib.Path(pool_csvs[(bench, fn_name)])
        n_rows, n_nonfinite = None, None
        if pool_csv.exists():
            pool = pd.read_csv(pool_csv)
            if "visited" in pool.columns:
                pool = pool[pool["visited"] == 1]
            cost = pd.to_numeric(pool["cost"], errors="coerce")
            n_rows = int(len(pool))
            n_nonfinite = int((~np.isfinite(cost)).sum())
        rows.append(
            {
                "benchmark": bench,
                "fn_name": fn_name,
                "n_requested": stats["requested"],
                "n_accepted": stats["accepted"],
                "n_rows": n_rows,
                "n_timed_out": stats["timed_out"],
                "n_timed_out_in_pool": n_nonfinite,
                "n_failed": stats["failed"],
                "n_over_budget": stats["over_budget"],
                "space_size": stats["space_size"],
                "sampling_mode": stats["sampling_mode"],
            }
        )
    return _write(pd.DataFrame(rows), out_csv, missing)


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
    offline_atim: pathlib.Path | None = None,
) -> bool:
    """search_roots: benchmark -> B4 dump root (holding
    infer_{fn}/seed_{k}/timings.csv); space_jsons: (benchmark, fn) ->
    space.json (its space_build_seconds is RQ2's setup-time row, available
    since B0 -- so rq2.csv has rows before any search has run).

    offline_atim contributes the baseline's side of "what a search costs":
    `tuning_wallclock_s` from the interchange CSV, as `search_wallclock_s`
    under whatever system the row names. Only schedules tuned on this
    machine carry one -- the published ones were tuned on the authors'
    hardware, so they have no wall clock that compares to ours and are
    dropped here rather than borrowed from a reproduced run."""
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

    if offline_atim is not None:
        offline = _offline_rows(offline_atim, "atim")
        if offline is None:
            missing.append(f"offline ATiM rows ({offline_atim})")
        else:
            tuned_here = offline.dropna(subset=["tuning_wallclock_s"])
            if tuned_here.empty:
                missing.append(f"ATiM tuning wall clock ({offline_atim})")
            for _, row in tuned_here.iterrows():
                rows.append(
                    {
                        "benchmark": row["benchmark"],
                        "fn_name": row["fn_name"],
                        "system": row["system"],
                        "seed": None,
                        "n_candidates": None,
                        "search_wallclock_s": float(row["tuning_wallclock_s"]),
                        "space_build_s": None,
                    }
                )
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
        offline = _offline_rows(path, system)
        if offline is None:
            missing.append(f"offline {system} rows ({path})")
        else:
            frames.append(offline)
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
