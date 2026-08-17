"""Transcribe ATiM's scheduled TIR into our configuration space.

    python points/transcribe_atim.py [--traces points/traces]
                                    [--only mtv_4MB mtv_64MB ...]

Writes points/{atim_published,atim_reproduced}/{bench}.json from the
`*.tir.py` dumps beside each trace, following the format and the reading
procedure in points/README.md.

The TIR is the input rather than the `apply_trace_*` decision list, because
the facts we need are already resolved there: `T.thread_binding(16,
thread="blockIdx.x")` states the extent and the axis, `T.axis.spatial(1024,
i_0 * 64 + ...)` states the iteration extent and which loops index it, and
the `*_local` staging blocks state the WRAM tile as their own loop bounds.
Reading the decision list means re-deriving all of that by replaying splits,
reorders and rfactors.

This generates; it does not certify. `doit compile_points
invariants_report` is the check -- feasible-set membership rejects a
configuration that does not exist, and the invariants report names the axis
that was misread when it does exist but moves different data. Points that
this script gets wrong are meant to be corrected in the JSON by hand, so
each one records how it was derived and every assumption it made.

What it cannot know, it does not guess:

- Which block axis is our outermost workgroup axis. ATiM binds blockIdx.x
  and blockIdx.y; their DPU linearization is not stated in the TIR, so when
  two dimensions are distributed across the block axes both readings are
  emitted as candidates and measured. That is what candidates are for.
- Operations ATiM has no counterpart for. Its schedules fuse the scaling of
  GEMV into the kernel, while our IR keeps it as a separate op that must be
  distributed over the same workgroup. Its tiling is derived from the
  workgroup shape, and reported as an assumption.
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
EVAL_DIR = HERE.parent
DATA_DIR = EVAL_DIR / "data"

# "mtv_4096_1_4096.published.tir.py"
TIR_NAME = re.compile(r"^(\w+?)_(\d+)_(\d+)_(\d+)\.(\w+)\.tir\.py$")
# "mram tile size of iteration dim M (extent 1024) of cinm.op.gemv; ..." -- the
# op is named only when the function has more than one, so it is read off the
# parameter name (`gemv.M.mram`) and the doc supplies the dim and its extent.
PARAM_DOC = re.compile(r"iteration dim (\w+) \(extent (\d+)\)")

# ATiM names a task by its extents; our functions are named after the size of
# the largest operand, which is what these compute. int64 for red (its
# accumulator type), int32 everywhere else -- see atim_eval.eval_mod.
LARGEST_OPERAND = {
    "va": lambda m, n, k: m * 4,
    "geva": lambda m, n, k: m * 4,
    "red": lambda m, n, k: m * 8,
    "mtv": lambda m, n, k: m * k * 4,
    "gemv": lambda m, n, k: m * k * 4,
    "ttv": lambda m, n, k: m * n * k * 4,
    "mmtv": lambda m, n, k: m * n * k * 4,
}


def size_label(nbytes: int) -> str:
    """4194304 -> "4MB", the suffix our function names carry."""
    return f"{nbytes // 2**20}MB"


# ── reading the TIR ─────────────────────────────────────────────────────────


class Loop:
    __slots__ = ("var", "extent", "thread")

    def __init__(self, var: str, extent: int, thread: str | None):
        self.var, self.extent, self.thread = var, extent, thread


class Block:
    def __init__(self, name: str, enclosing: list[Loop]):
        self.name = name
        self.enclosing = enclosing
        # (name, role, extent, {loop vars the index expression reads})
        self.axes: list[tuple[str, str, int, set[str]]] = []


def _const(node) -> int | None:
    return node.value if isinstance(node, ast.Constant) else None


def _kwarg(call: ast.Call, name: str):
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _call_attr(node) -> str | None:
    """ "T.axis.spatial(...)" -> "axis.spatial"; "T.grid(...)" -> "grid"."""
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return None
    parts = [node.func.attr]
    inner = node.func.value
    while isinstance(inner, ast.Attribute):
        parts.append(inner.attr)
        inner = inner.value
    return ".".join(reversed(parts))


def parse_tir(path: pathlib.Path) -> list[Block]:
    """Every block of the module, each carrying the loops enclosing it."""
    tree = ast.parse(path.read_text())
    blocks: list[Block] = []

    def walk(node, loops: list[Loop]):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.For):
                walk_for(child, loops)
            elif isinstance(child, ast.With):
                walk_with(child, loops)
            else:
                walk(child, loops)

    def walk_for(node: ast.For, loops: list[Loop]):
        kind = _call_attr(node.iter)
        args = node.iter.args if isinstance(node.iter, ast.Call) else []
        extents = [_const(a) for a in args]
        if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name):
            kind = node.iter.func.id  # range(n)
        thread = None
        if kind == "thread_binding":
            axis = _kwarg(node.iter, "thread")
            thread = _const(axis)
        names = (
            [t.id for t in node.target.elts]
            if isinstance(node.target, ast.Tuple)
            else [node.target.id]
        )
        # T.grid(a, b) binds one name per extent; every other form binds one.
        added = [
            Loop(name, extents[i] if i < len(extents) else None, thread)
            for i, name in enumerate(names)
        ]
        walk(node, loops + [loop for loop in added if loop.extent is not None])

    def walk_with(node: ast.With, loops: list[Loop]):
        name = None
        for item in node.items:
            if _call_attr(item.context_expr) == "block":
                name = _const(item.context_expr.args[0])
        if name is None:
            walk(node, loops)
            return
        block = Block(name, loops)
        blocks.append(block)
        extent_of = {loop.var: loop.extent for loop in loops}
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign):
                continue
            attr = _call_attr(stmt.value)
            if attr == "axis.remap":
                # `vi_0, vi_1_0 = T.axis.remap("SS", [i_0, i_1_0])` -- one axis
                # per loop, taking that loop's extent and its kind from the
                # string. A partial-sum index an rfactor introduced appears
                # this way, so skipping it loses a whole iteration dimension.
                kinds = _const(stmt.value.args[0]) or ""
                for kind, var in zip(kinds, stmt.value.args[1].elts):
                    if not isinstance(var, ast.Name) or var.id not in extent_of:
                        continue
                    block.axes.append(
                        (
                            var.id,
                            "reduce" if kind == "R" else "spatial",
                            extent_of[var.id],
                            {var.id},
                        )
                    )
                continue
            if attr not in ("axis.spatial", "axis.reduce"):
                continue
            extent = _const(stmt.value.args[0])
            if extent is None:
                continue
            reads = {
                n.id for n in ast.walk(stmt.value.args[1]) if isinstance(n, ast.Name)
            }
            target = stmt.targets[0]
            block.axes.append(
                (
                    target.id if isinstance(target, ast.Name) else "?",
                    attr.split(".")[1],
                    extent,
                    reads,
                )
            )
        walk(node, loops)

    walk(tree, [])
    return blocks


# ── deriving the configuration ──────────────────────────────────────────────


class Derivation:
    """What one TIR says about the workgroup and the tiling, plus the notes a
    reader needs to check it."""

    def __init__(self):
        self.dpus = 1
        self.tasklets = 1
        self.axis_of_dim: dict[int, str] = {}  # dim index -> blockIdx.* axis
        self.mram: list[int] = []
        self.wram: list[int] = []
        self.extents: list[int] = []
        self.notes: list[str] = []


def compute_block(blocks: list[Block]) -> Block | None:
    """The block doing the arithmetic: inside the thread bindings, not a
    staging copy (`*_local`), not an initialiser, and the widest of those."""
    candidates = [
        b
        for b in blocks
        if b.axes
        and any(loop.thread for loop in b.enclosing)
        and not b.name.endswith(("_local", "_init", "_shared"))
    ]
    return max(candidates, key=lambda b: len(b.axes), default=None)


def group_axes(block: Block, extents: list[int]) -> list[list[int]] | None:
    """Assign the block's axes to the op's iteration dimensions, in order.

    An rfactor splits one dimension across two axes -- GEMV's K becomes a
    spatial axis of 16 (the partial-sum index, one per DPU) and a reduce axis
    of 64 -- so axes and dimensions are not one to one. Consume axes
    left to right until each dimension's extent is accounted for; the TIR
    keeps an rfactor axis adjacent to the axis it split from, so this is
    unambiguous where it succeeds and returns None where it does not.
    """
    groups, i = [], 0
    for extent in extents:
        product, group = 1, []
        while product < extent and i < len(block.axes):
            product *= block.axes[i][2]
            group.append(i)
            i += 1
        if product != extent:
            return None
        groups.append(group)
    return groups if i == len(block.axes) else None


def derive(blocks: list[Block], extents: list[int]) -> Derivation | str:
    """The configuration one TIR describes, or why it could not be read."""
    d = Derivation()
    d.extents = extents

    block = compute_block(blocks)
    if block is None:
        return "no compute block inside a thread binding"

    threads: dict[str, Loop] = {}
    for loop in block.enclosing:
        if loop.thread:
            threads[loop.var] = loop
    for loop in threads.values():
        if loop.thread.startswith("blockIdx"):
            d.dpus *= loop.extent
        elif loop.thread.startswith("threadIdx"):
            d.tasklets *= loop.extent

    groups = group_axes(block, extents)
    if groups is None:
        got = [(a[1], a[2]) for a in block.axes]
        return f"cannot match block axes {got} to iteration extents {extents}"

    # The MRAM tile is what one (dpu, tasklet) leaf is left holding: the
    # dimension's extent divided by however much of it the thread bindings
    # spread across the workgroup.
    for gi, group in enumerate(groups):
        spread = 1
        for ai in group:
            for var in block.axes[ai][3]:
                if var in threads:
                    spread *= threads[var].extent
                    if threads[var].thread.startswith("blockIdx"):
                        d.axis_of_dim[gi] = threads[var].thread
        extent = extents[gi]
        if extent % spread:
            return f"dim {gi} of extent {extent} is spread over {spread} workers"
        d.mram.append(extent // spread)

    # The WRAM tile is what a staging block copies in one go: its own loop
    # bounds, which appear as the `ax*` loops it is nested in. Operands stage
    # different subsets, so a dimension takes the largest tile staged of it,
    # and one no operand stages is held whole.
    #
    # A staged axis belongs to the dimension it shares loop variables with,
    # not to the one of equal extent: GEMV over a square matrix has two
    # dimensions of the same extent staged as different tiles.
    indexed_by = [set().union(*(block.axes[ai][3] for ai in group)) for group in groups]
    staged: dict[int, int] = {}
    for other in blocks:
        if not other.name.endswith("_local") or not other.axes:
            continue
        inner = {
            loop.var: loop.extent
            for loop in other.enclosing
            if loop.var.startswith("ax")
        }
        for _, _, _, reads in other.axes:
            overlap = [len(reads & vars_) for vars_ in indexed_by]
            if not any(overlap):
                continue
            gi = overlap.index(max(overlap))
            tile = 1
            for var in reads:
                if var in inner:
                    tile *= inner[var]
            staged[gi] = max(staged.get(gi, 0), tile)
    for gi in range(len(extents)):
        tile = staged.get(gi, d.mram[gi])
        # A staged tile larger than the leaf holds means the extent matched
        # the wrong operand; the leaf bounds it either way.
        d.wram.append(min(tile, d.mram[gi]) or 1)
    if not staged:
        d.notes.append("no staging block found; WRAM tiles assumed to be the MRAM tile")

    return d


# ── mapping onto our parameters ─────────────────────────────────────────────


def read_space(space_json: pathlib.Path) -> dict:
    """Our parameters for one function, grouped into the ops they tile."""
    space = json.loads(space_json.read_text())
    ops: dict[str, list[tuple[str, int]]] = collections.OrderedDict()
    order_params: dict[str, dict] = {}
    others: list[str] = []
    for param in space["params"]:
        hit = PARAM_DOC.search(param.get("doc", ""))
        if hit and param["name"].endswith(".mram"):
            op = param["name"].split(".")[0]
            ops.setdefault(op, []).append((hit.group(1), int(hit.group(2))))
        elif param["name"].endswith(".order"):
            order_params[param["name"].split(".")[0]] = param
        elif param["name"] not in ("dpus", "tasklets") and not param["name"].endswith(
            ".wram"
        ):
            others.append(param["name"])
    return {"ops": ops, "orders": order_params, "others": others, "raw": space}


def transcribe(tir_path: pathlib.Path, space: dict, task: tuple) -> tuple[list, list]:
    """(candidates, problems) for one TIR under one function's space."""
    blocks = parse_tir(tir_path)

    # The op ATiM scheduled is the one whose iteration extents its TIR
    # matches; any other op of the function has no counterpart in the trace.
    derived, primary, problems = None, None, []
    for op, dims in space["ops"].items():
        result = derive(blocks, [extent for _, extent in dims])
        if isinstance(result, Derivation):
            derived, primary = result, op
            break
        problems.append(f"as cinm.op.{op}: {result}")
    if derived is None:
        return [], problems

    params: dict[str, int] = {"dpus": derived.dpus, "tasklets": derived.tasklets}
    leaves = derived.dpus * derived.tasklets
    notes = list(derived.notes)

    for (dim, _), mram, wram in zip(space["ops"][primary], derived.mram, derived.wram):
        params[f"{primary}.{dim}.mram"] = mram
        params[f"{primary}.{dim}.wram"] = wram

    # Ops ATiM folded into its kernel still have to be distributed over the
    # same workgroup, which fixes their tiling given the workgroup shape.
    for op, dims in space["ops"].items():
        if op == primary:
            continue
        for dim, extent in dims:
            if len(dims) == 1 and extent % leaves == 0:
                tile = extent // leaves
            else:
                tile = max(1, extent // leaves)
                problems.append(
                    f"cinm.op.{op} dim {dim} (extent {extent}) cannot be spread over"
                    f" {leaves} leaves; the space has no configuration with"
                    f" dpus*tasklets = {leaves} for this function"
                )
            params[f"{op}.{dim}.mram"] = tile
            params[f"{op}.{dim}.wram"] = tile
            notes.append(
                f"cinm.op.{op} has no counterpart in ATiM's schedule (it fuses the"
                f" operation into its kernel); {dim} tiled to fill the workgroup"
            )

    for name in space["others"]:
        if name.startswith("fuse."):
            # Fusing constrains the producer to whole output tiles, which a
            # K-split kernel does not produce.
            params[name] = 1
            notes.append(f"{name}=1 (unfused): ATiM's K-split leaves partial sums")

    problems += verify(params, space)

    # One candidate per reading of the workgroup axis order.
    orderings = _order_candidates(space, derived)
    candidates = []
    for label, assignment, why in orderings:
        candidates.append(
            {
                "label": label,
                "params": {**params, **assignment},
                "expected": {"D": [derived.dpus], "T": [derived.tasklets]},
                "notes": "; ".join(notes + ([why] if why else [])),
            }
        )
    return candidates, problems


def verify(params: dict, space: dict) -> list[str]:
    """The space's arithmetic constraints, checked here so a point that
    cannot exist says so now rather than at compile time.

    Only the ones that need nothing but the parameters: tiles divide, and
    each op's leaves fill the workgroup exactly. Capacity (MRAM/WRAM per
    tasklet) needs the operand layout and is left to the solver.
    """
    leaves = params["dpus"] * params["tasklets"]
    problems = []
    for op, dims in space["ops"].items():
        product = 1
        for dim, extent in dims:
            mram = params[f"{op}.{dim}.mram"]
            wram = params[f"{op}.{dim}.wram"]
            if extent % mram:
                problems.append(f"{op}.{dim}.mram={mram} does not divide {extent}")
            if mram % wram:
                problems.append(f"{op}.{dim}.wram={wram} does not divide {mram}")
            product *= extent // mram if mram else 0
        if product != leaves:
            problems.append(
                f"cinm.op.{op} covers {product} leaves, not dpus*tasklets = {leaves}"
            )
    return problems


def _order_candidates(space: dict, derived: Derivation) -> list[tuple[str, dict, str]]:
    """Readings of which dimension owns which workgroup axis.

    ATiM's TIR says a dimension sits on blockIdx.x or blockIdx.y; it does not
    say which of those is the slower-varying DPU index, and our order
    parameter is defined by that. With more than one dimension distributed
    the two readings are different configurations, so both are measured.
    """
    if not space["orders"]:
        return [("r0", {}, "")]
    name, param = next(iter(space["orders"].items()))
    orderings = param.get("orderings", [])
    distributed = {gi for gi in derived.axis_of_dim}
    if len(distributed) < 2:
        # One distributed dimension (or none): every ordering describes the
        # same workgroup, so the space's first is as good as any.
        return [("r0", dict(orderings[0]["assignment"]), "")]
    return [
        (
            f"r{i}",
            dict(o["assignment"]),
            f"{name}={o['order']} is one reading of ATiM's blockIdx axes;"
            f" their DPU linearization is not stated in the TIR",
        )
        for i, o in enumerate(orderings)
    ]


# ── driver ──────────────────────────────────────────────────────────────────


def space_json(bench: str, fn: str) -> pathlib.Path:
    return DATA_DIR / bench / "space" / f"infer_{fn}" / "space.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--traces", type=pathlib.Path, default=HERE / "traces")
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="FN_NAME",
        help="fn_names to transcribe, for iterating on a few points; every"
        " fn_name of a bench you want kept in its file has to be named",
    )
    args = parser.parse_args()
    only = set(args.only or ())
    unmatched = set(only)

    by_source: dict[str, dict[str, list]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    problems: list[str] = []
    seen = 0

    for tir in sorted(args.traces.glob("*.tir.py")):
        hit = TIR_NAME.match(tir.name)
        if not hit:
            continue
        op, m, n, k, label = (
            hit.group(1),
            int(hit.group(2)),
            int(hit.group(3)),
            int(hit.group(4)),
            hit.group(5),
        )
        if op not in LARGEST_OPERAND:
            problems.append(f"{tir.name}: unknown operator {op}")
            continue
        bench = f"prim_{op}"
        fn = f"{op}_{size_label(LARGEST_OPERAND[op](m, n, k))}"
        if only and fn not in only:
            continue
        unmatched.discard(fn)
        sj = space_json(bench, fn)
        if not sj.exists():
            problems.append(f"{fn}: no space.json ({sj}); run `doit space`")
            continue

        seen += 1
        space = read_space(sj)
        candidates, why = transcribe(tir, space, (op, m, n, k))
        problems += [f"{fn}: {w}" for w in why]
        if not candidates:
            continue
        by_source[f"atim_{label}"][bench].append(
            {
                "fn_name": fn,
                "trace": str(tir.relative_to(EVAL_DIR)),
                "candidates": candidates,
            }
        )

    # A name no trace answers to would otherwise be silent, and the file it was
    # meant to keep a point in gets written without that point.
    problems += [
        f"{fn}: --only names it, but no trace produces it" for fn in sorted(unmatched)
    ]

    written = 0
    for source, benches in sorted(by_source.items()):
        for bench, points in sorted(benches.items()):
            out = HERE / source / f"{bench}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            points.sort(key=lambda p: p["fn_name"])
            out.write_text(json.dumps(points, indent=2) + "\n")
            written += 1
            fns = ", ".join(p["fn_name"] for p in points)
            print(f"{out.relative_to(EVAL_DIR)}: {len(points)} points ({fns})")

    print(f"\n{seen} TIR dumps read, {written} files written")
    if problems:
        print(f"\n{len(problems)} could not be transcribed as they stand:")
        for p in problems:
            print(f"  {p}")
    print(
        "\nNothing here is checked. Run `doit compile_points invariants_report`:"
        " an infeasible configuration is rejected by the space, and the"
        " invariants report names the axis a compiled one misreads."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
