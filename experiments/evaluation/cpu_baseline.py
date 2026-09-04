"""fig:wholeprogram's CPU context bar: the RQ4 workloads compiled for the
host CPU by stock TVM.

RQ4's headline is UPMEM-vs-UPMEM (per-operator vs whole-program allocation).
This produces the third, non-competing bar: what the same program costs on
the host the DPUs are attached to, so a reader can place both arms against
something they already have a feel for.

Which TVM: ATiM's `evaluation/tvm_cputest` checkout -- stock TVM 0.13, the
one ATiM used for its own CPU baseline. Using the same compiler means our
context bar and ATiM's published CPU numbers are the same measurement, and
it needs no install: it is already built. Point $CPU_TVM_HOME elsewhere to
override. Its ctypes layer reads `np.float_`, removed in NumPy 2.0, so this
re-execs under an interpreter with NumPy < 2 (ATiM ships `atim-venv`), the
same bootstrap dance as points/_atim_env.py.

What is held to be a one-time cost, and why it is fair:

  - Weights are passed as `params`, so relay.build binds them as constants
    and folds the x86 layout transform into the built module. A serving
    deployment packs its weights once; charging that per inference would be
    the mirror of the sin RQ4 exists to expose, on the other machine.
  - `dense(a, b)` computes `a @ b.T`, so a weight is stored transposed --
    again a layout choice a deployment makes once, for data it owns.
  - The activation is a runtime input, set before timing and never
    re-uploaded: on the CPU there is no upload, which is exactly the point
    of the bar. The DPU arms pay scatter and load; the CPU pays neither and
    still has to read every weight from DRAM on every inference, which is
    what its bar shows.

Tuning: meta_schedule over one module holding all four programs, one
database per size class (see tune_class). --trials is the budget for that
class; 0 builds with TVM's fallback schedules in seconds.

How much the budget buys, measured on the bench machine (2x Xeon Silver
4216), is a strong function of size class, and it is worth knowing before
spending an afternoon on it. At 1MB the weights are cache-resident and the
kernel is compute-bound: 64 trials is ~50 s and worth 1.5-1.8x. At 256MB
the program streams 512 MB from DRAM per inference at ~15 GB/s, and 256
trials (~13 min) move it under 2% -- with an activation only 8 rows tall
there is no schedule that makes a weight cheaper to read once. The device
arms are the ones RQ4 argues about; the CPU bar just needs to be honest,
so the budget used is recorded in the CSV rather than left to be assumed.
"""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import shutil
import subprocess
import sys
import time

REEXEC_FLAG = "_CPU_TVM_REEXEC"

# Stock-TVM checkouts to try, in order: ATiM's CPU submodule first (see the
# module docstring), then its main tree, which is the same TVM with the
# UPMEM backend added -- fine for CPU codegen if the submodule is not built.
TVM_GUESSES = (
    pathlib.Path.home() / "Work/atim/evaluation/tvm_cputest",
    pathlib.Path(__file__).resolve().parents[2]
    / "third-party/atim/evaluation/tvm_cputest",
    pathlib.Path.home() / "Work/atim",
)
PYTHON_GUESSES = (
    pathlib.Path.home() / "miniconda3/envs/atim-venv/bin/python",
    pathlib.Path.home() / "anaconda3/envs/atim-venv/bin/python",
)

# Size class -> N, the side of each N x N i32 weight. The source of truth is
# experiments/{2mm,3mm}*.mlir; keep the two in step (1MB = 512^2 * 4 B).
CLASSES = {"1MB": 512, "16MB": 2048, "64MB": 4096, "256MB": 8192}
D = 8  # activation rows, as in the workloads


def find_tvm(explicit: str | None = None) -> pathlib.Path:
    candidates = [pathlib.Path(explicit)] if explicit else []
    if os.environ.get("CPU_TVM_HOME"):
        candidates.append(pathlib.Path(os.environ["CPU_TVM_HOME"]))
    candidates += list(TVM_GUESSES)
    for path in candidates:
        path = path.expanduser()
        if (path / "python/tvm/__init__.py").exists() and (
            path / "build/libtvm.so"
        ).exists():
            return path.resolve()
    raise SystemExit(
        "no built TVM checkout found; pass --tvm or set $CPU_TVM_HOME. Tried: "
        + ", ".join(str(c) for c in candidates)
    )


def bootstrap(tvm_flag: str | None, python: str | None) -> pathlib.Path:
    """The TVM checkout, re-execing this script under an interpreter that can
    import it when the ambient one cannot (NumPy 2.0, missing tvm, ...)."""
    tvm_home = find_tvm(tvm_flag)
    try:
        import tvm  # noqa: F401

        return tvm_home
    except Exception:
        if os.environ.get(REEXEC_FLAG):
            raise

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tvm_home / "python")]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env["TVM_LIBRARY_PATH"] = str(tvm_home / "build")
    env[REEXEC_FLAG] = "1"

    candidates = [python] if python else []
    if os.environ.get("CPU_TVM_PYTHON"):
        candidates.append(os.environ["CPU_TVM_PYTHON"])
    candidates += [str(p) for p in PYTHON_GUESSES]
    for candidate in candidates:
        exe = shutil.which(candidate) if candidate else None
        if exe is None and candidate and pathlib.Path(candidate).expanduser().exists():
            exe = str(pathlib.Path(candidate).expanduser())
        if exe is None:
            continue
        probe = subprocess.run(
            [exe, "-c", "import tvm"], env=env, capture_output=True, text=True
        )
        if probe.returncode:
            print(f"{exe}: cannot import TVM, skipping", file=sys.stderr)
            continue
        print(f"re-running under {exe}", file=sys.stderr)
        os.execve(exe, [exe, str(pathlib.Path(__file__).resolve()), *sys.argv[1:]], env)
    raise SystemExit(
        "no interpreter could import TVM; pass --python or set $CPU_TVM_PYTHON."
        " It needs NumPy < 2.0."
    )


# ─── The workloads ───────────────────────────────────────────────────────────
#
# One builder per program, mirroring experiments/<program>.mlir. Each returns
# (relay function, {param name: transposed weight}, {input name: array},
# reference), where the reference wraps in uint32 exactly like the device
# kernels' i32 (see experiments/bench/common.hpp): these chains overflow i32
# by construction, and the wrap is part of what both sides compute.


def _programs(np, relay, tvm, n: int, dt: str = "int32", prefix: str = ""):
    def rnd(*shape):
        return np.random.randint(0, 50, size=shape).astype(dt)

    def var(name, rows, cols):
        return relay.var(prefix + name, shape=(rows, cols), dtype=dt)

    def mm(a, b):  # a @ b, in uint32 wraparound
        return (a.astype(np.uint32) @ b.astype(np.uint32)).astype(dt)

    def dense(a, b):  # relay's a @ b.T
        return relay.nn.dense(a, b, out_dtype=dt)

    def build(fn_args, out, params, inputs, ref):
        # Weights go in transposed: dense multiplies by b.T, and storing the
        # transpose is a deployment-time choice about data it owns.
        return {
            "args": fn_args,
            "out": out,
            "func": relay.Function(fn_args, out),
            "params": {prefix + k: tvm.nd.array(v.T.copy()) for k, v in params.items()},
            "inputs": {prefix + k: tvm.nd.array(v) for k, v in inputs.items()},
            "ref": ref,
        }

    def two_mm_seq():
        X, W1, W2 = rnd(D, n), rnd(n, n), rnd(n, n)
        x, w1, w2 = var("x", D, n), var("w1", n, n), var("w2", n, n)
        return build(
            [x, w1, w2],
            dense(dense(x, w1), w2),
            {"w1": W1, "w2": W2},
            {"x": X},
            [mm(mm(X, W1), W2)],
        )

    def two_mm_par():
        X, W1, W2 = rnd(D, n), rnd(n, n), rnd(n, n)
        x, w1, w2 = var("x", D, n), var("w1", n, n), var("w2", n, n)
        return build(
            [x, w1, w2],
            relay.Tuple([dense(x, w1), dense(x, w2)]),
            {"w1": W1, "w2": W2},
            {"x": X},
            [mm(X, W1), mm(X, W2)],
        )

    def three_mm_parseq():
        X, W1, W2, W3 = rnd(D, n), rnd(n, n), rnd(n, n), rnd(n, n)
        x = var("x", D, n)
        w1, w2, w3 = var("w1", n, n), var("w2", n, n), var("w3", n, n)
        return build(
            [x, w1, w2, w3],
            relay.Tuple([dense(dense(x, w1), w2), dense(x, w3)]),
            {"w1": W1, "w2": W2, "w3": W3},
            {"x": X},
            [mm(mm(X, W1), W2), mm(X, W3)],
        )

    def three_mm_diamond():
        # j = (A@B) @ (C@D), B and C static. The right branch is computed
        # transposed -- (C@D).T = D.T @ C.T = dense(D.T, C) -- so the join is
        # dense(l, r.T) and nothing transposes at run time. D.T is supplied
        # as the input, the same host-side layout choice as a weight's.
        A, B, C, Dt = rnd(D, n), rnd(n, n), rnd(n, n), rnd(D, n)
        a, b = var("a", D, n), var("b", n, n)
        c, dt_ = var("c", n, n), var("dt", D, n)
        left = dense(a, b)
        right_t = dense(dt_, c)  # (C @ D).T
        return build(
            [a, b, c, dt_],
            dense(left, right_t),
            # c enters dense(dt_, c) already in the orientation it is used,
            # so undo build()'s transpose for it.
            {"b": B, "c": C.T},
            {"a": A, "dt": Dt},
            [mm(mm(A, B), mm(C, Dt.T))],
        )

    return {
        "2mm_seq": two_mm_seq,
        "2mm_par": two_mm_par,
        "3mm_parseq": three_mm_parseq,
        "3mm_diamond": three_mm_diamond,
    }


# ─── Build, verify, measure ──────────────────────────────────────────────────


def run_one(np, tvm, relay, ms, graph_executor, prog, target, database, repeat: int):
    """(latency ms, verified) for one built program."""
    mod = tvm.IRModule.from_expr(prog["func"])
    if database is None:
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=prog["params"])
    else:
        lib = ms.relay_integration.compile_relay(database, mod, target, prog["params"])

    dev = tvm.cpu(0)
    module = graph_executor.GraphModule(lib["default"](dev))
    for name, value in prog["inputs"].items():
        module.set_input(name, value)

    module.run()  # warm up, and produce the outputs the check reads
    verified = all(
        np.array_equal(module.get_output(i).numpy(), want)
        for i, want in enumerate(prog["ref"])
    )
    result = module.module.time_evaluator("run", dev, number=3, repeat=repeat)()
    return float(result.mean) * 1e3, verified


def tune_class(ms, tvm, relay, programs, target, work_dir, trials, per_iter):
    """One meta_schedule database per size class, tuned over all four
    programs at once.

    Tuning them one call at a time would spend the budget on the same kernel
    shapes four times over -- within a class the programs are built from the
    same two dense shapes, and a fresh tune_relay call re-tunes a workload
    whether or not the database already holds records for it. Tuning a single
    module holding every program's output instead lets extract_tasks dedupe
    the shapes first, so the budget goes to distinct kernels and each
    program's own build then finds them in the database.
    """
    args, outs, params = [], [], {}
    for prog in programs:
        args += prog["args"]
        outs.append(prog["out"])
        params.update(prog["params"])
    combined = tvm.IRModule.from_expr(relay.Function(args, relay.Tuple(outs)))
    work_dir.mkdir(parents=True, exist_ok=True)
    return ms.relay_integration.tune_relay(
        mod=combined,
        params=params,
        target=target,
        work_dir=str(work_dir),
        max_trials_global=trials,
        num_trials_per_iter=min(per_iter, trials),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tvm", help="TVM checkout ($CPU_TVM_HOME)")
    parser.add_argument("--python", help="interpreter with TVM ($CPU_TVM_PYTHON)")
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent / "results/cpu.csv",
    )
    parser.add_argument(
        "--work-dir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent / "data/cpu_tuning",
        help="meta_schedule databases, one directory per size class",
    )
    parser.add_argument("--classes", nargs="*", default=list(CLASSES))
    parser.add_argument("--programs", nargs="*", default=None)
    parser.add_argument(
        "--trials",
        type=int,
        default=256,
        help="meta_schedule trial budget per size class (0 = untuned)",
    )
    parser.add_argument("--trials-per-iter", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--num-cores",
        type=int,
        default=os.cpu_count() // 2 if os.cpu_count() else 32,
        help="physical cores TVM schedules for; also sets TVM_NUM_THREADS",
    )
    parser.add_argument(
        "--mcpu",
        default="cascadelake",
        help="LLVM -mcpu (cascadelake = the bench machine's Xeon Silver 4216)",
    )
    args = parser.parse_args()

    bootstrap(args.tvm, args.python)
    os.environ.setdefault("TVM_NUM_THREADS", str(args.num_cores))

    import numpy as np
    import tvm
    from tvm import meta_schedule as ms
    from tvm import relay
    from tvm.contrib import graph_executor

    target = tvm.target.Target(f"llvm -mcpu={args.mcpu} --num-cores={args.num_cores}")
    rows = []
    for cls in args.classes:
        if cls not in CLASSES:
            raise SystemExit(f"unknown size class {cls!r}; know {list(CLASSES)}")
        n = CLASSES[cls]
        names = args.programs or list(_programs(np, relay, tvm, n))
        # Per-program variable prefixes: the tuning module holds all four at
        # once, and two programs' "x" must not collide there.
        programs = {
            name: _programs(np, relay, tvm, n, prefix=f"{name}_")[name]()
            for name in names
        }

        database = None
        if args.trials:
            t0 = time.time()
            database = tune_class(
                ms,
                tvm,
                relay,
                list(programs.values()),
                target,
                args.work_dir / cls,
                args.trials,
                args.trials_per_iter,
            )
            print(f"[{cls}] tuned in {time.time() - t0:.0f}s", flush=True)

        for name, prog in programs.items():
            ms_time, verified = run_one(
                np, tvm, relay, ms, graph_executor, prog, target, database, args.repeat
            )
            print(
                f"[{cls}] {name}: {ms_time:.3f} ms"
                f"{'' if verified else '  *** MISMATCH ***'}",
                flush=True,
            )
            rows.append(
                {
                    "program": name,
                    "cls": cls,
                    "n": n,
                    "total_ms": round(ms_time, 4),
                    "verified": int(verified),
                    "trials": args.trials,
                    "num_cores": args.num_cores,
                    "target": str(target),
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")
    return 0 if all(r["verified"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
