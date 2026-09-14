#!/usr/bin/env python3
from __future__ import annotations
import os
import shlex
import subprocess
from pathlib import Path
from typing import Sequence

from . import torch_frontend as torch_nb

def _find_repo_root(start: Path) -> Path:
    for p in (start, *start.parents):
        if (p / ".git").is_dir():
            return p
    raise RuntimeError(f"Could not locate repository root from {start}")

_REPO_ROOT = _find_repo_root(Path.cwd().resolve())
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
_COMPILE_SH = _SCRIPTS_DIR / "docker_llvm_compile.sh"
_RUN_SH = _SCRIPTS_DIR / "docker_run.sh"

def _run_script(cmd_argv: Sequence[str | os.PathLike], *, env: dict | None = None) -> str:
    # The scripts need the environment the notebook server was started in --
    # the pixi one -- which this process inherits and passes straight through.
    argv = [str(a) for a in cmd_argv]
    proc = subprocess.run(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env or os.environ.copy(),
        cwd=_REPO_ROOT,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {shlex.join(argv)}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return proc.stdout

def _rel_to_repo(p: str | os.PathLike) -> str:
    ap = Path(p).resolve()
    try:
        return str(ap.relative_to(_REPO_ROOT))
    except ValueError:
        return str(ap)

def compile_binary(
    driver_c: str | os.PathLike,
    llvm_ll: str | os.PathLike,
    output_bin: str | os.PathLike,
    *,
    extra_env: dict | None = None,
) -> str:
    # Older toolchains inside the Docker image expect the legacy names for the
    # stack management intrinsics; rewrite them on the fly if needed.
    ll_path = Path(llvm_ll)
    if ll_path.is_file():
        text = ll_path.read_text()
        if "llvm.stacksave.p0" in text or "llvm.stackrestore.p0" in text:
            text = text.replace("llvm.stacksave.p0", "llvm.stacksave")
            text = text.replace("llvm.stackrestore.p0", "llvm.stackrestore")
            # Next to the original rather than over it, so the cell that
            # produced it can be re-run and compared.
            ll_path = ll_path.with_suffix(".legacy-intrinsics.ll")
            ll_path.write_text(text)
            llvm_ll = ll_path
    if not _COMPILE_SH.is_file():
        raise FileNotFoundError(f"Missing script: {_COMPILE_SH}")
    cmd = [_COMPILE_SH, _rel_to_repo(driver_c), _rel_to_repo(llvm_ll), _rel_to_repo(output_bin)]
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return _run_script(cmd, env=env)

def run_binary(
    binary: str | os.PathLike,
    *,
    extra_env: dict | None = None,
) -> str:
    if not _RUN_SH.is_file():
        raise FileNotFoundError(f"Missing script: {_RUN_SH}")
    cmd = [_RUN_SH, _rel_to_repo(binary)]
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return _run_script(cmd, env=env)

def build_and_run(
    driver_c: str | os.PathLike,
    llvm_ll: str | os.PathLike,
    output_bin: str | os.PathLike,
    *,
    extra_env: dict | None = None,
) -> tuple[str, str]:
    comp_out = compile_binary(driver_c, llvm_ll, output_bin, extra_env=extra_env)
    run_out = run_binary(output_bin, extra_env=extra_env)
    return comp_out, run_out


def print_output() -> str:
    # docker_run.sh writes gem5's output into an `out` directory beside the
    # binary it was given.
    path = torch_nb.BINARY_OUTPUT_DIR / "out" / "program.out"
    if not path.is_file():
        raise FileNotFoundError(f"program.out not found at {path}")
    return path.read_text(errors="replace")
