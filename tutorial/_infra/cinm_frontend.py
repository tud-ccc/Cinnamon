"""Helpers for running cinm-opt from the tutorial notebooks."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# Repository discovery
# ---------------------------------------------------------------------------

def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / ".git").is_dir():
            return candidate
    raise RuntimeError(f"Cannot locate repository root from {start}")


NOTEBOOK_CWD = Path.cwd().resolve()
REPO_ROOT = _find_repo_root(NOTEBOOK_CWD)
TUTORIAL_DIR = REPO_ROOT / "tutorial"


# ---------------------------------------------------------------------------
# Tool lookup
# ---------------------------------------------------------------------------

def _first_existing(paths: Iterable[Path]) -> Path | None:
    for candidate in paths:
        if candidate.is_dir() or candidate.is_file():
            return candidate.resolve()
    return None


CINM_BUILD = Path(os.environ.get("CINM_BUILD_DIR", REPO_ROOT / "build"))
CINM_BIN = _first_existing([CINM_BUILD / "bin"])


def _require_tool(tool: str) -> Path:
    override = os.environ.get(f"{tool.upper().replace('-', '_')}_PATH")
    if override:
        path = Path(override).expanduser().resolve()
        if path.is_file():
            return path
    if CINM_BIN:
        candidate = CINM_BIN / tool
        if candidate.is_file():
            return candidate.resolve()
    which = shutil.which(tool)
    if which:
        return Path(which).resolve()
    raise FileNotFoundError(
        f"{tool} not found. Build Cinnamon (build/bin/{tool}) or set {tool.upper().replace('-', '_')}_PATH."
    )


cinm_opt = _require_tool("cinm-opt")


# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------

def run(cmd: Sequence[os.PathLike[str] | str], *, stdin: str | bytes | None = None, cwd: Path | None = None) -> str:
    proc = subprocess.run(
        [str(Path(arg)) if isinstance(arg, Path) else str(arg) for arg in cmd],
        input=stdin.encode() if isinstance(stdin, str) else stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        text=True,
    )
    if proc.returncode:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(map(str, cmd))}\n{proc.stderr}"
        )
    return proc.stdout


__all__ = [
    "NOTEBOOK_CWD",
    "REPO_ROOT",
    "TUTORIAL_DIR",
    "cinm_opt",
    "run",
]
