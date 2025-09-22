#!/usr/bin/env python3
from __future__ import annotations
import os
import shlex
import subprocess
from pathlib import Path
from typing import Sequence

def _find_repo_root(start: Path) -> Path:
    for p in (start, *start.parents):
        if (p / ".git").is_dir():
            return p
    raise RuntimeError(f"Could not locate repository root from {start}")

_REPO_ROOT = _find_repo_root(Path.cwd().resolve())
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
_COMPILE_SH = _SCRIPTS_DIR / "docker_llvm_compile.sh"
_RUN_SH = _SCRIPTS_DIR / "docker_run.sh"
_ENV_SH = _REPO_ROOT / "env.sh"

def _run_with_env(cmd_argv: Sequence[str | os.PathLike], *, env: dict | None = None) -> str:
    # Run in a login shell that sources env.sh before executing the command
    # so PATH/LD_LIBRARY_PATH/etc from env.sh are in effect.
    cmd_str = " ".join(shlex.quote(str(Path(a)) if isinstance(a, (Path, os.PathLike)) else str(a))
                       for a in cmd_argv)
    if _ENV_SH.is_file():
        shell_line = f"source {shlex.quote(str(_ENV_SH))} >/dev/null 2>&1 && {cmd_str}"
    else:
        shell_line = cmd_str
    proc = subprocess.run(
        ["/bin/bash", "-lc", shell_line],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env or os.environ.copy(),
        cwd=_REPO_ROOT,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {cmd_str}\n"
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
    if not _COMPILE_SH.is_file():
        raise FileNotFoundError(f"Missing script: {_COMPILE_SH}")
    cmd = [_COMPILE_SH, _rel_to_repo(driver_c), _rel_to_repo(llvm_ll), _rel_to_repo(output_bin)]
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return _run_with_env(cmd, env=env)

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
    return _run_with_env(cmd, env=env)

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


from tutorial._infra import torch_frontend as torch_nb

def print_output() -> str:
    root = getattr(torch_nb, "ROOT_DIR", getattr(torch_nb, "REPO_ROOT", Path.cwd()))
    candidates = [
        root / "third-party" / "ALPINE" / "working_test" / "out" / "program.out",
    ]
    for p in candidates:
        if p.is_file():
            text = p.read_text(errors="replace")
            # print(text, end="")
            return text
    raise FileNotFoundError(f"program.out not found in: {candidates[0]} or {candidates[1]}")