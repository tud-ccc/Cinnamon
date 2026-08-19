"""What each ATiM schedule's host-to-device transfer actually costs.

    python points/atim_transfer_report.py [--traces points/traces]
                                          [--only gemv_1024_1_1024.reproduced ...]
                                          [--static-h2d .../reproduced/static_h2d.csv]
                                          [--csv points/traces/transfers.csv]

Reports, per operand of every `*.tir.py` dump, how many `dpu_push_xfer`
calls its transfer is broken into, and -- where ATiM's measurements are
available -- what that cost and the bandwidth it achieved.

The point is that bytes do not predict the cost and push count does. A
transfer is issued as `InitXfer` / `BindXfer` per DPU / `PushXfer`, and
`ExtractPimTransferSchedule` can collapse the whole operand into a single
push only when each DPU's MRAM image is a contiguous slice of the host
buffer. When the schedule permutes the host and device index order the DPU
image is a transposed view, no contiguous mapping exists past the innermost
agreeing axis, and the pass falls back to pushing that run -- sometimes two
int32 -- once per iteration of everything outside it. Each push then costs a
fixed ~0.5 ms at 2048 DPUs (2048 `dpu_prepare_xfer` calls plus rank
orchestration) no matter how few bytes it carries, so a schedule can pay
seconds to move the same 256 MiB another moves in 21 ms.

That matters to E1 twice over. It is the largest single source of variation
between two ATiM schedules that agree on every tiling factor, and ATiM
excludes it: an operand in `pragma_explicit_h2d` is lifted into its own
`copy_<symbol>` function and charged as one-time weight residency, so none
of it appears in the reported H2D/Kernel/D2H. A transcription measured
against those numbers has to account for the same transfer the same way, or
it loses on bookkeeping rather than on code.

The push counts come from replaying ATiM's own lowering: the passes below
are the ones `evaluation/base.py:pre_kernel` runs to produce its `split.py`
dump, stopping where the transfer intrinsics have been made explicit.

Needs ATiM's TVM and the UPMEM SDK, but no DPUs. See `_atim_env`.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import re
import sys

import _atim_env

HERE = pathlib.Path(__file__).resolve().parent

TIR_SUFFIX = ".tir.py"
# "gemv_1024_1_1024.reproduced" -> ("gemv", 1024, 1, 1024, "reproduced"), the
# key static_h2d.csv records a measurement under.
STEM = re.compile(r"^(\w+?)_(\d+)_(\d+)_(\d+)\.(\w+)$")
# "A_1" and "A" are the same operand; TVM suffixes on rebinding.
DEDUP = re.compile(r"_\d+$")


def split_ir(mod):
    """ATiM's lowering, up to where transfers are explicit intrinsics."""
    import tvm
    from tvm.target import Target
    from tvm.tir.transform import (
        AnnotateDeviceRegions,
        AnnotateEntryFunc,
        BindTarget,
        ExtractPimTransferSchedule,
        SplitHostDevice,
        SplitPimTransfer,
        Simplify,
        VerifyMemory,
    )

    config = {"tir.UpmemUseDummyKernel": False, "tir.UpmemKernelOptimize": -1}
    with tvm.transform.PassContext(config=config):
        lowered = tvm.lower(mod)
        # canon_target_map_and_host rewrites the key, so the map cannot be
        # indexed with the target it was built from.
        by_target, _ = Target.canon_target_map_and_host(
            {Target("upmem"): lowered}, "llvm"
        )
        target = next(t for t in by_target if t.kind.name == "upmem")
        split = by_target[target]
        for one in (
            BindTarget(target),
            VerifyMemory(),
            AnnotateEntryFunc(),
            AnnotateDeviceRegions(),
            ExtractPimTransferSchedule(),
            SplitHostDevice(),
            SplitPimTransfer(),
            Simplify(),
        ):
            split = one(split)
    return split


def transfer_sites(func):
    """(operand, pushes, elems_per_push, direction) per transfer in `func`.

    One `dpu_parallel_transfer_init` is one `dpu_push_xfer`, so the pushes a
    site costs is the trip count of the loops around it. `post_order_visit`
    cannot carry that context, hence the explicit walk.
    """
    from tvm import tir

    sites = []

    def walk(stmt, trips):
        if isinstance(stmt, tir.For):
            walk(stmt.body, trips * int(stmt.extent))
            return
        if isinstance(stmt, tir.SeqStmt):
            for sub in stmt:
                walk(sub, trips)
            return
        for attr in ("body", "then_case", "else_case"):
            sub = getattr(stmt, attr, None)
            if isinstance(sub, tir.Stmt):
                walk(sub, trips)
        if isinstance(stmt, tir.Evaluate) and isinstance(stmt.value, tir.Call):
            call = stmt.value
            if "dpu_parallel_transfer_init" in str(call.op):
                operand = DEDUP.sub("", str(call.args[0].name))
                sites.append((operand, trips, int(call.args[2]), int(call.args[3])))

    walk(func.body, 1)
    return sites


def measurements(path: pathlib.Path | None) -> dict:
    """ATiM's per-operand transfer measurements, keyed as the trace stems are.

    Only the h2d side exists here: `static_h2d.csv` is written by the
    `h2d()` loop, which runs before the timed region.
    """
    if path is None or not path.exists():
        return {}
    out = {}
    for row in csv.DictReader(path.open()):
        key = (
            row["Workload"],
            int(row["M"]),
            int(row["N"]),
            int(row["K"]),
            row["ConfigLabel"],
            row["Operand"],
        )
        out[key] = {
            "bytes": int(row["Bytes"]),
            "scatter_ms": float(row["ScatterMs"]),
            "explicit": row["Explicit"] == "True",
        }
    return out


def rows_for(tir_path: pathlib.Path, measured: dict) -> list[dict]:
    module = _atim_env.load_module(tir_path)
    split = split_ir(module)
    stem = tir_path.name[: -len(TIR_SUFFIX)]
    hit = STEM.match(stem)
    task = (
        (
            hit.group(1),
            int(hit.group(2)),
            int(hit.group(3)),
            int(hit.group(4)),
            hit.group(5),
        )
        if hit
        else None
    )

    rows = []
    for gv, func in split.functions.items():
        name = gv.name_hint
        # The device kernel's own MRAM reads are not host transfers.
        if name.endswith("_kernel"):
            continue
        for operand, pushes, elems, direction in transfer_sites(func):
            row = {
                "stem": stem,
                "fn": name,
                "operand": operand,
                "dir": "h2d" if direction == 1 else "d2h",
                # An operand ATiM lifted out of the kernel; its cost is
                # excluded from the reported H2D.
                "lifted": name.startswith("copy_"),
                "pushes": pushes,
                "bytes_per_push_per_dpu": elems * 4,
            }
            found = measured.get(task + (operand,)) if task else None
            if found:
                row["bytes"] = found["bytes"]
                row["scatter_ms"] = found["scatter_ms"]
                # What the transfer achieved end to end, over the whole
                # operand rather than per DPU: the number that makes a
                # push-bound transfer obvious next to a bandwidth-bound one.
                row["mib_per_ms"] = (
                    found["bytes"] / 2**20 / found["scatter_ms"]
                    if found["scatter_ms"]
                    else None
                )
            rows.append(row)
    return rows


def render(rows: list[dict]) -> None:
    header = (
        f"{'trace':<32} {'operand':<12} {'dir':<4} {'lifted':<7} {'MiB':>8}"
        f" {'pushes':>7} {'B/push/DPU':>11} {'scatter ms':>11} {'MiB/ms':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        # `:.3g` rather than a fixed precision: operands here span a scalar
        # ALPHA and half a gibibyte, and rounding the small ones to 0 hides
        # that they were measured at all.
        mib = f"{row['bytes'] / 2**20:.3g}" if "bytes" in row else "-"
        ms = f"{row['scatter_ms']:.1f}" if "scatter_ms" in row else "-"
        bw = f"{row['mib_per_ms']:.2f}" if row.get("mib_per_ms") is not None else "-"
        print(
            f"{row['stem']:<32} {row['operand']:<12} {row['dir']:<4}"
            f" {'yes' if row['lifted'] else 'no':<7} {mib:>8}"
            f" {row['pushes']:>7} {row['bytes_per_push_per_dpu']:>11} {ms:>11} {bw:>9}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--traces", type=pathlib.Path, default=HERE / "traces")
    parser.add_argument(
        "--only", nargs="+", metavar="STEM", help="trace stems to report on"
    )
    parser.add_argument(
        "--static-h2d",
        type=pathlib.Path,
        help="ATiM's static_h2d.csv, for the measured columns"
        " (default: $ATIM_HOME/evaluation/reproduced/static_h2d.csv)",
    )
    parser.add_argument("--csv", type=pathlib.Path, help="also write the table here")
    _atim_env.add_arguments(parser)
    args = parser.parse_args()

    atim = _atim_env.bootstrap(__file__, args.atim, args.python)
    _atim_env.warn_without_sdk()

    static_h2d = args.static_h2d or atim / "evaluation/reproduced/static_h2d.csv"
    measured = measurements(static_h2d)
    if not measured:
        print(
            f"no measurements at {static_h2d}; reporting push counts only",
            file=sys.stderr,
        )

    only = set(args.only or ())
    unmatched = set(only)
    rows, problems = [], []
    for tir_path in sorted(args.traces.glob(f"*{TIR_SUFFIX}")):
        stem = tir_path.name[: -len(TIR_SUFFIX)]
        if only and stem not in only:
            continue
        unmatched.discard(stem)
        try:
            rows += rows_for(tir_path, measured)
        except Exception as exc:
            problems.append(f"{stem}: {type(exc).__name__}: {exc}".replace("\n", " "))

    # Slowest first, then by push count for the ones with no measurement --
    # a high push count is the finding even before it has been measured.
    rows.sort(key=lambda r: (-r.get("scatter_ms", -1), -r["pushes"]))
    render(rows)

    if args.csv:
        fields = [
            "stem",
            "fn",
            "operand",
            "dir",
            "lifted",
            "pushes",
            "bytes_per_push_per_dpu",
            "bytes",
            "scatter_ms",
            "mib_per_ms",
        ]
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwritten to {args.csv}")

    if problems:
        print(f"\n{len(problems)} could not be lowered:")
        for problem in problems:
            print(f"  {problem}")
    problems += [
        f"{stem}: --only names it, but no such trace" for stem in sorted(unmatched)
    ]
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
