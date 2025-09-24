"""Utilities for preparing torch-mlir based tutorial notebooks.

This module hides the repository/tool discovery logic so that the
notebooks can stay focused on the MLIR content.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# Repository + toolchain discovery
# ---------------------------------------------------------------------------

def _find_repo_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / ".git").is_dir():
            return path
    raise RuntimeError(f"Could not locate repository root from {start}")


NOTEBOOK_CWD = Path.cwd().resolve()
REPO_ROOT = _find_repo_root(NOTEBOOK_CWD)
TUTORIAL_DIR = REPO_ROOT / "tutorial"
ASSETS_DIR = TUTORIAL_DIR / "assets"
DRIVERS_DIR = ASSETS_DIR / "drivers"
BINARY_OUTPUT_DIR = REPO_ROOT / "third-party/ALPINE/working_test"

def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.is_dir():
            return path.resolve()
    return None


LLVM_ROOT = Path(os.environ.get("LLVM_ROOT", REPO_ROOT / "third-party" / "llvm"))
LLVM_BIN = _first_existing(
    [LLVM_ROOT / "build" / "bin", LLVM_ROOT / "install" / "bin"]
)
LLVM_LIB = _first_existing(
    [LLVM_ROOT / "build" / "lib", LLVM_ROOT / "install" / "lib"]
)

TORCH_MLIR_ROOT = Path(
    os.environ.get("TORCH_MLIR_ROOT", REPO_ROOT / "third-party" / "torch-mlir")
)
TORCH_MLIR_BIN = _first_existing(
    [TORCH_MLIR_ROOT / "build" / "bin", TORCH_MLIR_ROOT / "install" / "bin"]
)

# Torch-MLIR python packages are installed either under "install/python_packages"
# or exposed from the build tree. Ensure the package directory itself is on
# sys.path so ``import torch_mlir`` resolves to the in-tree build instead of any
# pip-installed stub.
TORCH_MLIR_PYTHON = _first_existing(
    [
        TORCH_MLIR_ROOT / "install" / "python_packages" / "torch_mlir",
        TORCH_MLIR_ROOT / "build" / "python_packages" / "torch_mlir",
    ]
)

if TORCH_MLIR_PYTHON and str(TORCH_MLIR_PYTHON) not in sys.path:
    sys.path.insert(0, str(TORCH_MLIR_PYTHON))

if LLVM_LIB:
    # Update the current process environment so that importing torch-mlir shared
    # libraries succeeds even outside the ``run`` helper.
    def _append_path(var: str) -> None:
        existing = os.environ.get(var)
        prefix = str(LLVM_LIB)
        if not existing:
            os.environ[var] = prefix
        elif prefix not in existing.split(os.pathsep):
            os.environ[var] = prefix + os.pathsep + existing

    _append_path("LD_LIBRARY_PATH")
    _append_path("DYLD_LIBRARY_PATH")


def _require_tool(tool: str) -> Path:
    override = os.environ.get(f"{tool.upper().replace('-', '_')}_PATH")
    if override:
        candidate = Path(override).expanduser().resolve()
        if candidate.is_file():
            return candidate
    search_dirs: Sequence[Path | None] = (
        TORCH_MLIR_BIN,
        LLVM_BIN,
        REPO_ROOT / "build" / "bin",
    )
    for directory in search_dirs:
        if directory:
            candidate = directory / tool
            if candidate.is_file():
                return candidate.resolve()
    which = shutil.which(tool)
    if which:
        return Path(which).resolve()
    raise FileNotFoundError(
        f"{tool} not found; set {tool.upper().replace('-', '_')}_PATH or build torch-mlir."
    )


_candidate_torch_mlir_opts = [
    TORCH_MLIR_ROOT / "install" / "bin" / "torch-mlir-opt",
    TORCH_MLIR_ROOT / "build" / "bin" / "torch-mlir-opt",
]
torch_mlir_opt = None
for candidate in _candidate_torch_mlir_opts:
    if candidate.is_file():
        torch_mlir_opt = candidate.resolve()
        break
if torch_mlir_opt is None:
    torch_mlir_opt = _require_tool("torch-mlir-opt")

# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in ("MLIR_PLUGINS_PATH", "MLIR_LOAD_PLUGINS", "MLIR_ENABLE_PLUGINS"):
        env.pop(key, None)
    lib_dirs = [d for d in (LLVM_LIB,) if d]
    if lib_dirs:
        joined = os.pathsep.join(str(d) for d in lib_dirs)
        env["LD_LIBRARY_PATH"] = joined + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        env["DYLD_LIBRARY_PATH"] = joined + os.pathsep + env.get("DYLD_LIBRARY_PATH", "")
    return env


def run(cmd: Sequence[os.PathLike[str] | str], *, stdin: str | bytes | None = None, cwd: Path | None = None) -> str:
    proc = subprocess.run(
        [str(arg) for arg in cmd],
        input=stdin.encode() if isinstance(stdin, str) else stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=_clean_env(),
    )
    if proc.returncode:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(map(str, cmd))}\n{proc.stderr.decode()}"
        )
    return proc.stdout.decode()


ARTIFACTS_DIR = Path(
    os.environ.get("CINNAMON_NOTEBOOK_ARTIFACTS", TUTORIAL_DIR / "artifacts")
)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def print_setup() -> None:
    print(f"Repository root -> {REPO_ROOT}")
    print(f"torch-mlir-opt  -> {torch_mlir_opt}")
    print(f"Artifacts dir   -> {ARTIFACTS_DIR}")


__all__ = [
    "ARTIFACTS_DIR",
    "NOTEBOOK_CWD",
    "REPO_ROOT",
    "print_setup",
    "run",
    "torch_mlir_opt",
    "TORCH_MLIR_PYTHON",
]
