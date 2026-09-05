"""Getting at ATiM's TVM from the evaluation's own tree.

Shared by `dump_atim_c.py` and `atim_transfer_report.py`, both of which have
to run passes from ATiM's compiler over the `*.tir.py` dumps in
`points/traces/`.

Nothing here is installable. ATiM's TVM is used out of its source tree --
`python/` on `PYTHONPATH`, `build/libtvm.so` via `TVM_LIBRARY_PATH` -- and
the UPMEM backend shells out to `dpu-upmem-dpurte-clang`, so `$UPMEM_HOME`
has to be on `PATH` too. On top of that its ctypes layer reads `np.float_`
at import, which NumPy 2.0 removed, so the evaluation's own venv cannot
import it at all. `bootstrap()` is what stops that from surfacing as a bare
`ModuleNotFoundError`: it finds the checkout, and re-execs the calling
script under an interpreter that can actually load it.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
import types

# Set on the child so a re-exec that still cannot import TVM raises the real
# import error instead of bouncing between interpreters.
REEXEC_FLAG = "_ATIM_ENV_REEXEC"

# Interpreters to try when the ambient one has no usable TVM, in order. ATiM
# ships an `atim-venv` conda environment; everything else is the user's to
# name, because a wrong guess here runs ATiM's passes against a different
# TVM and the answer would silently be someone else's.
PYTHON_GUESSES = (
    pathlib.Path.home() / "miniconda3/envs/atim-venv/bin/python",
    pathlib.Path.home() / "anaconda3/envs/atim-venv/bin/python",
)

ATIM_GUESSES = (
    pathlib.Path.home() / "Work/atim",
    pathlib.Path(__file__).resolve().parents[3] / "third-party/atim",
)


def add_arguments(parser) -> None:
    """The two flags every script sharing this bootstrap accepts."""
    parser.add_argument("--atim", help="ATiM checkout ($ATIM_HOME)")
    parser.add_argument("--python", help="interpreter with ATiM's TVM ($ATIM_PYTHON)")


def find_atim(explicit: str | None = None) -> pathlib.Path:
    """ATiM's checkout: the one holding both its Python tree and its build."""
    candidates = [pathlib.Path(explicit)] if explicit else []
    if os.environ.get("ATIM_HOME"):
        candidates.append(pathlib.Path(os.environ["ATIM_HOME"]))
    candidates += list(ATIM_GUESSES)
    for path in candidates:
        path = path.expanduser()
        if (path / "python/tvm/__init__.py").exists() and (
            path / "build/libtvm.so"
        ).exists():
            return path.resolve()
    raise SystemExit(
        "no ATiM checkout with a built libtvm.so found; pass --atim or set"
        f" $ATIM_HOME. Tried: {', '.join(str(c) for c in candidates)}"
    )


def child_env(atim: pathlib.Path) -> dict:
    """The environment ATiM's TVM needs: its Python tree, its shared library,
    and the UPMEM SDK the UPMEM backend shells out to."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(atim / "python")] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env["TVM_LIBRARY_PATH"] = str(atim / "build")
    upmem = env.get("UPMEM_HOME")
    if upmem:
        env["PATH"] = os.pathsep.join([f"{upmem}/bin", env.get("PATH", "")])
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [f"{upmem}/lib"]
            + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else [])
        )
    return env


def bootstrap(script: str, atim_flag: str | None = None, python: str | None = None):
    """Return ATiM's checkout, re-execing `script` first if TVM is missing.

    `script` is the caller's `__file__`; the re-exec has to land back in the
    script the user ran, not in this module.
    """
    atim = find_atim(atim_flag)
    try:
        import tvm  # noqa: F401

        return atim
    except Exception:
        if os.environ.get(REEXEC_FLAG):
            raise

    candidates = [python] if python else []
    if os.environ.get("ATIM_PYTHON"):
        candidates.append(os.environ["ATIM_PYTHON"])
    candidates += [str(p) for p in PYTHON_GUESSES]

    env = child_env(atim)
    env[REEXEC_FLAG] = "1"
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
            print(f"{exe}: cannot import ATiM's TVM, skipping", file=sys.stderr)
            continue
        print(f"re-running under {exe}", file=sys.stderr)
        resolved = str(pathlib.Path(script).resolve())
        os.execve(exe, [exe, resolved, *sys.argv[1:]], env)
    raise SystemExit(
        "no interpreter could import ATiM's TVM; pass --python or set"
        f" $ATIM_PYTHON. Tried: {', '.join(str(c) for c in candidates)}."
        " It needs NumPy < 2.0."
    )


def warn_without_sdk() -> None:
    """The UPMEM backend compiles what it generates, so a missing SDK fails
    the build rather than degrading it."""
    if not os.environ.get("UPMEM_HOME") and not shutil.which("dpu-upmem-dpurte-clang"):
        print(
            "warning: no dpu-upmem-dpurte-clang on PATH and $UPMEM_HOME unset;"
            " the UPMEM backend compiles the kernel it generates, so the build"
            " will fail before it hands anything back",
            file=sys.stderr,
        )


def load_module(path: pathlib.Path):
    """The IRModule a `*.tir.py` dump prints.

    The dump comments its own imports out (TVMScript prints them that way),
    and TVMScript's parser reaches back through `sys.modules` for the source
    file it is parsing, so both have to be supplied around the `exec`.
    """
    import tvm.script

    ns = types.ModuleType("atim_tir_dump")
    ns.__file__ = str(path)
    ns.I, ns.T = tvm.script.ir, tvm.script.tir
    sys.modules["atim_tir_dump"] = ns
    try:
        exec(compile(path.read_text(), str(path), "exec"), ns.__dict__)
    finally:
        del sys.modules["atim_tir_dump"]
    if not hasattr(ns, "Module"):
        raise ValueError("no `Module` in the dump")
    return ns.Module
