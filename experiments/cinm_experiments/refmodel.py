"""Price a dumped DPU program with the Python reference cost model
(third-party/cnm-cost-model/Predictor/cnmprog.py) and put the number next to
what the C++ cost model said about the very same program.

The dumps come from cinmopt.with_program_dump, which writes both halves of
the comparison into one directory: the <kernel>.cnmprog.json programs and the
cost.csv of the run that emitted them. Since the JSON carries no latencies --
only opcodes, dtypes, DMA sizes and trip counts -- a gap between the two
numbers is a difference in scheduling or in the calibrated tables, not in
what the two engines were asked to price.

Two things the comparison assumes, both true of the dumps as emitted today:

- cost.csv's kernel row for a compute block is the sum over that block's
  launches, and each launch dumps its own JSON, so the unit that can be
  compared is the whole dump directory, not a single file. A directory with
  any unpriced kernel therefore has no reference total at all -- summing the
  rest would silently understate it.
- a launch inside a host-side loop is charged once per iteration in cost.csv
  but dumped once, so such a program would compare as if the loop ran once.
"""

from __future__ import annotations

import csv
import dataclasses
import pathlib
import re
import subprocess

from .paths import ROOT, python_bin

PREDICTOR_DIR = ROOT / "third-party" / "cnm-cost-model" / "Predictor"
CNMPROG = PREDICTOR_DIR / "cnmprog.py"

# cnmprog.py's one line of output: "<kernel>: 5.621308 ms  (extrapolated)".
_MS_RE = re.compile(r":\s*([0-9.eE+-]+)\s*ms")


DUMP_SUFFIX = ".cnmprog.json"


@dataclasses.dataclass
class KernelPrice:
    """What the reference model made of one dumped program."""

    kernel: str  # the upmem.dpu_program symbol the dump is named after
    path: pathlib.Path
    ms: float | None  # None if the reference model would not price it
    error: str = ""


def price(
    json_path: pathlib.Path,
    *,
    exact: bool = False,
    timeout_s: float = 900,
    python: str | None = None,
) -> KernelPrice:
    """Run cnmprog.py over one dump. A refusal is a result, not an exception:
    the model rejects programs it has no honest price for (an instruction with
    no LUT entry, or the C++ builder's per-tasklet predication), and which
    programs those are is part of what the cross-check measures.

    `exact` selects the fully-unrolled oracle over the extrapolating engine --
    materialises one entry per dispatched instruction, so it is only viable
    for small programs."""
    json_path = pathlib.Path(json_path)
    kernel = json_path.name.removesuffix(DUMP_SUFFIX)
    cmd = [python or python_bin(), str(CNMPROG), str(json_path.resolve())]
    if exact:
        cmd.append("--exact")
    try:
        # cwd: cnmprog.py imports its siblings by bare module name. Its LUT
        # paths are absolute (derived from __file__), so only the imports care.
        r = subprocess.run(
            cmd,
            cwd=str(PREDICTOR_DIR),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return KernelPrice(kernel, json_path, None, f"timeout after {timeout_s}s")

    if r.returncode != 0:
        lines = [ln.strip() for ln in r.stderr.splitlines() if ln.strip()]
        return KernelPrice(kernel, json_path, None, lines[-1] if lines else "failed")
    m = _MS_RE.search(r.stdout)
    if not m:
        return KernelPrice(
            kernel, json_path, None, f"unparseable output: {r.stdout.strip()!r}"
        )
    return KernelPrice(kernel, json_path, float(m.group(1)))


def cpp_kernel_ms(costs_csv: pathlib.Path) -> float | None:
    """The C++ cost model's kernel time over every compute block in a
    costs_csv, which is the side of the breakdown the dumped programs stand
    for -- transfers are host-side and have no counterpart in a dump.
    launchOverhead is a kernel-category row too, and is left out: it is
    charged per launch by the host model, not computed from the program."""
    costs_csv = pathlib.Path(costs_csv)
    if not costs_csv.exists():
        return None
    total = 0.0
    with open(costs_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row["category"] == "kernel" and row["label"] != "launchOverhead":
                total += float(row["cost_ms"])
    return total


@dataclasses.dataclass
class Comparison:
    """Both engines' verdict on one dump directory's programs."""

    name: str
    dump_dir: pathlib.Path
    cpp_ms: float | None
    kernels: list[KernelPrice]

    @property
    def unpriced(self) -> list[KernelPrice]:
        return [k for k in self.kernels if k.ms is None]

    @property
    def ref_ms(self) -> float | None:
        """The reference total, or None if any kernel went unpriced -- a
        partial sum is not comparable to cpp_ms (see the module docstring)."""
        if not self.kernels or self.unpriced:
            return None
        return sum(k.ms for k in self.kernels)

    @property
    def ratio(self) -> float | None:
        """reference / C++. 1.0 is agreement; >1 means the reference model
        prices the program higher."""
        if self.ref_ms is None or not self.cpp_ms:
            return None
        return self.ref_ms / self.cpp_ms


def compare(
    dump_dir: pathlib.Path, *, name: str | None = None, **price_kwargs
) -> Comparison | None:
    """Price every dump in a with_program_dump directory and pair the total
    with the cost.csv beside them. None if the directory holds no dumps (not
    compiled since the dumping was added)."""
    dump_dir = pathlib.Path(dump_dir)
    dumps = sorted(dump_dir.glob(f"*{DUMP_SUFFIX}"))
    if not dumps:
        return None
    return Comparison(
        name=name or dump_dir.parent.name,
        dump_dir=dump_dir,
        cpp_ms=cpp_kernel_ms(dump_dir / "cost.csv"),
        kernels=[price(d, **price_kwargs) for d in dumps],
    )


def comparison_rows(comparisons: list[Comparison]) -> list[dict]:
    """One flat row per kernel, for a CSV. Only ref_kernel_ms is per-kernel:
    the C++ side is only available as the total the kernel contributes to, so
    the two totals repeat down a multi-kernel config's rows."""
    return [
        {
            "name": c.name,
            "kernel": k.kernel,
            "ref_kernel_ms": k.ms,
            "ref_total_ms": c.ref_ms,
            "cpp_total_ms": c.cpp_ms,
            "ratio": c.ratio,
            "error": k.error,
        }
        for c in comparisons
        for k in c.kernels
    ]
