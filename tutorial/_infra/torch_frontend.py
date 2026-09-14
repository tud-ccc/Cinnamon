"""Utilities for preparing torch-mlir based tutorial notebooks.

This module hides the repository/tool discovery logic so that the
notebooks can stay focused on the MLIR content.
"""

from __future__ import annotations

import os
import shutil
import subprocess
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

ARTIFACTS_DIR = Path(
    os.environ.get("CINNAMON_NOTEBOOK_ARTIFACTS", TUTORIAL_DIR / "artifacts")
)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Cross-compiled binaries, and whatever gem5 writes beside them. Deliberately
# outside third-party/ALPINE, which is a submodule.
BINARY_OUTPUT_DIR = ARTIFACTS_DIR / "bin"
BINARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.is_dir():
            return path.resolve()
    return None


# Either the prebuilt LLVM that build-llvm.sh downloads, one built from the
# submodule, or whatever LLVM_BUILD_DIR points at -- the same three cases
# common.sh resolves between.
_LLVM_DIRS = [
    Path(os.environ[var])
    for var in ("LLVM_BUILD_DIR", "LLVM_ROOT")
    if os.environ.get(var)
] + [
    REPO_ROOT / "third-party" / "llvm-prebuilt",
    REPO_ROOT / "third-party" / "llvm" / "build",
]
LLVM_ROOT = _first_existing(_LLVM_DIRS)
LLVM_BIN = _first_existing([d / "bin" for d in _LLVM_DIRS])
LLVM_LIB = _first_existing([d / "lib" for d in _LLVM_DIRS])

TORCH_MLIR_ROOT = Path(
    os.environ.get("TORCH_MLIR_ROOT", REPO_ROOT / "third-party" / "torch-mlir")
)
TORCH_MLIR_BIN = _first_existing(
    [TORCH_MLIR_ROOT / "build" / "bin", TORCH_MLIR_ROOT / "install" / "bin"]
)

# torch_mlir is imported from the environment build-torch.sh installs it into,
# and the tools below find their libraries through their own RPATHs.


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
    "BINARY_OUTPUT_DIR",
    "DRIVERS_DIR",
]
