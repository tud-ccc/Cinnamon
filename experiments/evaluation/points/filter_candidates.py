"""Keep the transcription candidates whose kernels move what ATiM's moves.

    python points/filter_candidates.py [--only gemv_4MB ...] [--prune]

A point's candidates are the readings of a trace the transcriber could not
tell apart (points/README.md): ATiM's TIR says a dimension sits on
blockIdx.x or blockIdx.y, not which is the slower-varying DPU index. The
readings agree on every tiling factor, so `expected` (D, T) cannot separate
them -- but they assign different dimensions to the DPU axes, so each DPU
ends up holding a different slice of each operand. That footprint is stated
by both sides' compiled kernels:

- ATiM's: the `__mram` declarations of `points/traces/{stem}.dpu.c`
  (regenerate with `python points/dump_atim_c.py`), one sized array per
  operand per DPU.
- ours: the transfers of each candidate's compiled `lowered.mlir`, read by
  the same extraction `doit invariants_report` uses.

A candidate is faithful iff the multiset of per-DPU operand footprints
matches, scattered and gathered compared separately. Comparing multisets
rather than named pairs needs no operand correspondence, and the case that
motivated candidates in the first place -- GEMV over a square matrix, where
r0 hands every DPU the whole vector and r1 a slice -- differs exactly there
(32 KiB against 2 KiB of B per DPU).

Scalar transfers (<= 8 bytes per DPU) are ignored on both sides: ATiM keeps
its scalars in `__host` variables outside MRAM, while we broadcast them, so
they never correspond; and at 8 bytes they discriminate nothing.

By default this only reports. `--prune` rewrites the points JSON keeping
the matching candidates of every point where the match is *selective* --
some but not all candidates match. A point where every candidate matches
has nothing to prune, and one where none matches is left whole and reported
loudly: that is a mistranscription (or a stale compile), not a choice.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
EVAL_DIR = HERE.parent
DATA_DIR = EVAL_DIR / "data"

sys.path.insert(0, str(EVAL_DIR))
import invariants  # noqa: E402  (sibling module of the evaluation dir)

# `__mram_noinit int32_t A[4096];` in an ATiM kernel: one operand's per-DPU
# slice. `__host` scalars deliberately do not match.
_ATIM_DECL = re.compile(r"__mram(?:_noinit)?\s+(\w+)\s+(\w+)\[(\d+)\]")
# The two spellings of an MRAM write in ATiM's generated C: a DMA writeback
# or a direct store.
_ATIM_WRITE = re.compile(
    r"mram_write\([^;]*__mram_ptr void\*\)\s*\(\s*(\w+)|^\s*(\w+)\[[^\]]*\]\s*=", re.M
)

_ELT_BYTES = {
    "int8_t": 1,
    "uint8_t": 1,
    "int16_t": 2,
    "uint16_t": 2,
    "int32_t": 4,
    "uint32_t": 4,
    "float": 4,
    "int64_t": 8,
    "uint64_t": 8,
    "double": 8,
}

# Per-DPU footprints this small are scalars (ATiM: __host, ours: an 8-byte
# broadcast) and never correspond between the two sides.
_SCALAR_BYTES = 8

TRANSCRIBED_SYSTEM = {
    "atim_published": "atim_published_transcribed",
    "atim_reproduced": "atim_reproduced_transcribed",
}


def atim_footprints(
    dpu_c: pathlib.Path,
) -> tuple[collections.Counter, collections.Counter]:
    """(scattered, gathered) per-DPU byte footprints of ATiM's kernel.

    An operand the kernel writes is gathered; the rest are scattered. One
    both read and written would appear in both multisets on our side but
    only once here -- no ATiM schedule in the traces does that (their
    accumulators zero-init in WRAM), so it is not modelled.
    """
    text = dpu_c.read_text()
    written = {name for pair in _ATIM_WRITE.findall(text) for name in pair if name}
    scattered, gathered = collections.Counter(), collections.Counter()
    for c_type, name, count in _ATIM_DECL.findall(text):
        nbytes = _ELT_BYTES[c_type] * int(count)
        if nbytes <= _SCALAR_BYTES:
            continue
        (gathered if name in written else scattered)[nbytes] += 1
    return scattered, gathered


def candidate_footprints(
    lowered: pathlib.Path,
) -> tuple[collections.Counter, collections.Counter]:
    """(scattered, gathered) per-DPU byte footprints of one compiled
    candidate, via the invariants extraction."""
    facts = invariants.extract(lowered)
    num_dpus = facts["sets"][0][0] if facts["sets"] else 1
    scattered, gathered = collections.Counter(), collections.Counter()
    for name, tr in facts["transfers"].items():
        kind = name.split()[0]
        # A broadcast's total_bytes is the one host buffer (see
        # invariants.extract), which is exactly what every DPU receives; the
        # per-DPU kinds record wire bytes and divide back out.
        per_dpu_bytes = (
            tr["total_bytes"] if kind == "broadcast" else tr["total_bytes"] // num_dpus
        )
        if per_dpu_bytes <= _SCALAR_BYTES:
            continue
        if kind in ("scatter_on_array", "scatter_blocks", "broadcast"):
            scattered[per_dpu_bytes] += 1
        elif kind in ("gather_from_array", "gather_blocks"):
            gathered[per_dpu_bytes] += 1
    return scattered, gathered


def _fmt(counter: collections.Counter) -> str:
    return (
        "{"
        + ", ".join(
            f"{b}B x{n}" if n > 1 else f"{b}B" for b, n in sorted(counter.items())
        )
        + "}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="+", metavar="FN_NAME")
    parser.add_argument(
        "--prune",
        action="store_true",
        help="rewrite the points JSON keeping only matching candidates,"
        " where the match is selective",
    )
    args = parser.parse_args()
    only = set(args.only or ())

    problems: list[str] = []
    pruned = 0

    for source, system in TRANSCRIBED_SYSTEM.items():
        for points_json in sorted((HERE / source).glob("prim_*.json")):
            bench = points_json.stem
            points = json.loads(points_json.read_text())
            changed = False

            for point in points:
                fn = point["fn_name"]
                if only and fn not in only:
                    continue
                trace = EVAL_DIR / point["trace"]
                dpu_c = trace.with_name(trace.name.replace(".tir.py", ".dpu.c"))
                if not dpu_c.exists():
                    problems.append(f"{fn}: no {dpu_c.name}; run points/dump_atim_c.py")
                    continue
                want = atim_footprints(dpu_c)

                verdicts = []
                for cand in point["candidates"]:
                    label = cand["label"]
                    lowered = (
                        DATA_DIR
                        / bench
                        / f"points_{source}"
                        / "compiled"
                        / system
                        / fn
                        / label
                        / "lowered.mlir"
                    )
                    if not lowered.exists():
                        verdicts.append((label, None, None))
                        continue
                    got = candidate_footprints(lowered)
                    verdicts.append((label, got == want, got))

                matches = [label for label, ok, _ in verdicts if ok]
                print(
                    f"{system}/{fn}: ATiM scatters {_fmt(want[0])},"
                    f" gathers {_fmt(want[1])}"
                )
                for label, ok, got in verdicts:
                    if ok is None:
                        print(f"  {label}: not compiled -- cannot judge")
                    elif ok:
                        print(f"  {label}: MATCH")
                    else:
                        print(
                            f"  {label}: mismatch -- scatters {_fmt(got[0])},"
                            f" gathers {_fmt(got[1])}"
                        )

                # A single candidate is not a choice: a mismatch there is a
                # structural difference between the two lowerings (red, for
                # one: ATiM combines its tasklets on-DPU and gathers one
                # scalar, ours gathers every tasklet's partial), not a
                # misreading this filter could resolve.
                if len(point["candidates"]) < 2:
                    continue
                if not matches and all(ok is not None for _, ok, _ in verdicts):
                    problems.append(
                        f"{fn} ({source}): no candidate matches ATiM's kernel;"
                        " the transcription itself (or the compile) is wrong"
                    )
                if args.prune and matches and len(matches) < len(point["candidates"]):
                    point["candidates"] = [
                        c for c in point["candidates"] if c["label"] in matches
                    ]
                    changed = True
                    pruned += 1
                    print(f"  pruned to: {', '.join(matches)}")

            if changed:
                points_json.write_text(json.dumps(points, indent=2) + "\n")
                print(f"wrote {points_json.relative_to(EVAL_DIR)}")

    if pruned:
        print(f"\n{pruned} points pruned")
    if problems:
        print(f"\n{len(problems)} problems:")
        for p in problems:
            print(f"  {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
