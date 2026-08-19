"""Recompute a transcription's invariants from its compiled artifact.

A manually-transcribed configuration (an ATiM schedule, a CINM 1.0 rule
decision) is only trustworthy if the program we compile from it moves the
same data the same way the transcriber believes it does. This module reads
the upmem-level lowered.mlir of one candidate and extracts what can be
cross-checked without running anything: the DPU-set shapes (D, T), the
tasklet count of each program, per-transfer element counts and total
bytes, and the WRAM footprint per tasklet. report() prints these next to
the transcriber's own `expected` values and flags disagreements.

The extraction is textual (regexes over the ops' assembly formats), which
is deliberate: it runs on the artifact the pipeline actually produced,
needs no bindings, and if an assembly format changes the regexes miss --
report() then prints nothing for that field rather than something wrong.
"""

from __future__ import annotations

import pathlib
import re

_HIERARCHY = re.compile(r"upmem\.alloc_dpus\s*:\s*!upmem\.hierarchy<(\d+)x(\d+)>")
_TASKLETS = re.compile(r"upmem\.dpu_program\s+@\S+?\(\)\s+tasklets\((\d+)\)")
# An optional discardable-attribute dict between the operands and the type,
# e.g. the {upmem.timing_tag = "static:3"} the transfer ops carry.
_ATTRS = r"(?:\s*\{[^}]*\})?"
# upmem.scatter_blocks %x[32 elts, #map, 8 blocks] onto @buf of %h :
#   memref<...xi32> onto !upmem.hierarchy<256x1>
_BLOCKS_XFER = re.compile(
    r"upmem\.(scatter_blocks|gather_blocks)\s+\S+\[(\d+) elts, #\S+, (\d+) blocks\]"
    r"\s+(?:onto|from)\s+(@\S+)\s+of\s+\S+" + _ATTRS + r"\s*:\s*memref<[^>]*[iuf](\d+)>"
    r"\s+(?:onto|from)\s+!upmem\.hierarchy<(\d+)x(\d+)>"
)
# upmem.scatter_on_array %x[128 elts, #map] onto @buf of %h : ...
_ARRAY_XFER = re.compile(
    r"upmem\.(scatter_on_array|gather_from_array)\s+\S+\[(\d+) elts, #\S+\]"
    r"\s+(?:onto|from)\s+(@\S+)\s+of\s+\S+" + _ATTRS + r"\s*:\s*memref<[^>]*[iuf](\d+)>"
    r"\s+(?:onto|from)\s+!upmem\.hierarchy<(\d+)x(\d+)>"
)
# upmem.broadcast %t onto @buf of %h : memref<...> onto !upmem.hierarchy<DxT>
_BROADCAST = re.compile(
    r"upmem\.broadcast\s+\S+\s+onto\s+(@\S+)\s+of\s+\S+"
    + _ATTRS
    + r"\s*:\s*memref<([0-9x]+)x[iuf](\d+)[^>]*>\s+onto\s+!upmem\.hierarchy<(\d+)x(\d+)>"
)
_WRAM_ALLOCA = re.compile(
    r"memref\.alloca\(\)\s*:\s*memref<([0-9x]+)x[iuf](\d+),\s*#upmem\.wram>"
)


def _num_elements(dims: str) -> int:
    n = 1
    for d in dims.split("x"):
        n *= int(d)
    return n


def extract(lowered_mlir: pathlib.Path) -> dict:
    """The checkable invariants of one compiled candidate, as a dict:
    sets [(D, T), ...], tasklets [...], transfers
    {"<kind> <bufsym>": {per_dpu_elts, total_bytes}}, and
    wram_bytes_per_tasklet (the sum of per-tasklet WRAM allocas -- shared
    buffers are counted once per tasklet here, so treat it as an upper
    bound when tiles are thread-shared)."""
    text = pathlib.Path(lowered_mlir).read_text()

    sets = [(int(d), int(t)) for d, t in _HIERARCHY.findall(text)]
    tasklets = [int(t) for t in _TASKLETS.findall(text)]

    transfers = {}
    for kind, elts, blocks, buf, width, d, _t in _BLOCKS_XFER.findall(text):
        per_dpu = int(elts) * int(blocks)
        transfers[f"{kind} {buf}"] = {
            "per_dpu_elts": per_dpu,
            "total_bytes": per_dpu * int(d) * int(width) // 8,
        }
    for kind, elts, buf, width, d, _t in _ARRAY_XFER.findall(text):
        transfers[f"{kind} {buf}"] = {
            "per_dpu_elts": int(elts),
            "total_bytes": int(elts) * int(d) * int(width) // 8,
        }
    for buf, dims, width, d, _t in _BROADCAST.findall(text):
        n = _num_elements(dims)
        transfers[f"broadcast {buf}"] = {
            "per_dpu_elts": n,
            # One host-side buffer, replicated to every DPU by the runtime;
            # host bytes is the honest per-operand number.
            "total_bytes": n * int(width) // 8,
        }

    wram = sum(
        _num_elements(dims) * int(width) // 8
        for dims, width in _WRAM_ALLOCA.findall(text)
    )
    return {
        "sets": sets,
        "tasklets": tasklets,
        "transfers": transfers,
        "wram_bytes_per_tasklet": wram or None,
    }


def report(lowered_mlir: pathlib.Path, expected: dict) -> str:
    """Human-readable computed-vs-expected report. `expected` keys the
    transcriber may provide: D, T (ints), scattered_bytes / gathered_bytes
    (total over all scatter/gather transfers), wram_bytes_per_tasklet.
    Anything omitted is only reported, not checked."""
    inv = extract(lowered_mlir)
    lines = [f"computed from {lowered_mlir.name}:"]
    for d, t in inv["sets"]:
        lines.append(f"  set: {d} DPUs x {t} tasklets")
    for t in inv["tasklets"]:
        lines.append(f"  program tasklets: {t}")
    scattered = gathered = 0
    for name, tr in sorted(inv["transfers"].items()):
        lines.append(
            f"  {name}: {tr['per_dpu_elts']} elts/DPU, {tr['total_bytes']} B total"
        )
        if name.startswith(("scatter", "broadcast")):
            scattered += tr["total_bytes"]
        else:
            gathered += tr["total_bytes"]
    lines.append(f"  scattered total: {scattered} B, gathered total: {gathered} B")
    if inv["wram_bytes_per_tasklet"]:
        lines.append(f"  WRAM/tasklet (upper bound): {inv['wram_bytes_per_tasklet']} B")

    def check(name: str, want, got) -> str:
        ok = "ok" if want == got else "MISMATCH"
        return f"  {name}: expected {want}, computed {got}  [{ok}]"

    if expected:
        lines.append("against the transcriber's expectations:")
        if "D" in expected:
            lines.append(check("D", expected["D"], [d for d, _ in inv["sets"]]))
        if "T" in expected:
            lines.append(check("T", expected["T"], [t for _, t in inv["sets"]]))
        if "scattered_bytes" in expected:
            lines.append(
                check("scattered_bytes", expected["scattered_bytes"], scattered)
            )
        if "gathered_bytes" in expected:
            lines.append(check("gathered_bytes", expected["gathered_bytes"], gathered))
        if "wram_bytes_per_tasklet" in expected:
            lines.append(
                check(
                    "wram_bytes_per_tasklet",
                    expected["wram_bytes_per_tasklet"],
                    inv["wram_bytes_per_tasklet"],
                )
            )
    return "\n".join(lines) + "\n"
