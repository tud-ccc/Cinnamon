#!/usr/bin/env python3
"""Helper utilities imported by the tutorial notebooks.

The notebooks expect a small toolkit that knows how to
 - locate the MLIR/LLVM binaries that were built as part of the project,
 - run those tools while capturing their textual output, and
 - provide a writable directory for generated artifacts.

This module is intentionally lightweight so it can run in the
Jupyter environment started via ``start-notebook.sh``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Optional, Tuple  # Optional kept; added Tuple
import platform  # for OS detection

# The tutorial directory (``.../tutorial``) contains the notebooks,
# assets, and this helper module. The project root is one level above.
_TUTORIAL_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _TUTORIAL_DIR.parent

# Well-known binary locations relative to the repository root. These match
# the layout produced by the workflow scripts in ``.github/workflows``.
_KNOWN_BIN_SUBDIRS: Sequence[Path] = (
    _REPO_ROOT / "third-party" / "llvm" / "build" / "bin",
    _REPO_ROOT / "third-party" / "llvm" / "install" / "bin",
    _REPO_ROOT / "third-party" / "torch-mlir" / "build" / "bin",
    _REPO_ROOT / "third-party" / "torch-mlir" / "install" / "bin",
    _REPO_ROOT / "build" / "bin",  # fallback for local builds
)

# Environment variables that may point at a toolchain directory.
_ENV_HINTS: Sequence[str] = (
    "MLIR_BIN_DIR",
    "LLVM_BIN_DIR",
    "LLVM_BUILD_DIR",
    "LLVM_INSTALL_DIR",
    "TORCH_MLIR_BIN_DIR",
    "TORCH_MLIR_BUILD_DIR",
    "TORCH_MLIR_INSTALL_DIR",
)


def _normalize_dirs(values: Iterable[Path]) -> List[Path]:
    """Expand each directory hint to include ``dir`` and ``dir/bin``.

    ``LLVM_BUILD_DIR`` style variables often point at the build prefix,
    so we pessimistically add both variants.
    """

    result: List[Path] = []
    for path in values:
        if not path:
            continue
        path = path.resolve()
        if path not in result:
            result.append(path)
        bin_dir = path / "bin"
        if bin_dir.exists() and bin_dir not in result:
            result.append(bin_dir)
    return result


@lru_cache(maxsize=None)
def _candidate_tool_dirs() -> List[Path]:
    """Return an ordered list of directories to search for binaries."""

    env_dirs = [Path(os.environ[var]) for var in _ENV_HINTS if os.environ.get(var)]
    env_dirs = _normalize_dirs(env_dirs)

    # Directories from PATH (keep order).
    path_dirs = [Path(p) for p in os.environ.get("PATH", "").split(os.pathsep) if p]

    # Combine: environment hints first, then known repo locations, then PATH.
    combined: List[Path] = []
    for collection in (env_dirs, list(_KNOWN_BIN_SUBDIRS), path_dirs):
        for directory in collection:
            if directory.exists() and directory not in combined:
                combined.append(directory.resolve())
    return combined


def _find_tool(executable: str) -> Path | None:
    """Locate ``executable`` by consulting known directories and PATH."""

    # Quick path: respect explicit overrides: e.g., CLANG_PATH=/usr/bin/clang
    override = os.environ.get(executable.upper().replace("-", "_") + "_PATH")
    if override:
        override_path = Path(override).expanduser().resolve()
        if override_path.is_file():
            return override_path

    # Use shutil.which in case the environment already knows about it.
    which = shutil.which(executable)
    if which:
        return Path(which).resolve()

    for directory in _candidate_tool_dirs():
        candidate = directory / executable
        if candidate.is_file():
            return candidate
    return None


@lru_cache(maxsize=None)
def tools() -> Dict[str, Path]:
    """Return a mapping of tool names to resolved paths.

    Only tools that can be found are included in the mapping.
    """

    desired = [
        "mlir-opt",
        "mlir-translate",
        "mlir-cpu-runner",
        "mlir-cuda-runner",
        "mlir-runner",
        "clang",
        "opt",
        "llc",
    ]
    return {name: path for name in desired if (path := _find_tool(name))}


def mlir_translate(input_mlir: Path | str,
                   output: Path | str | None = None,
                   *,
                   extra_args: Sequence[str] = ()) -> Path:
    """Run ``mlir-translate`` on ``input_mlir`` and return the output path.

    By default this converts LLVM-dialect MLIR into textual LLVM IR.  You can
    pass additional flags through ``extra_args`` for other translations.
    """

    tool = tools().get("mlir-translate")
    if not tool:
        raise FileNotFoundError(
            "mlir-translate not found; ensure LLVM was built or set MLIR_BIN_DIR")

    input_mlir = Path(input_mlir).resolve()
    if output is None:
        output = input_mlir.with_suffix(".ll")
    output = Path(output).resolve()

    cmd: List[str] = [str(tool)]
    if extra_args:
        cmd.extend(str(arg) for arg in extra_args)
    else:
        # Default translation is LLVM IR emission.
        cmd.append("--mlir-to-llvmir")
    cmd.extend([str(input_mlir), "-o", str(output)])

    run(cmd)
    return output


def mlir_opt_path() -> Path:
    """Return the path to ``mlir-opt`` or raise a helpful error."""

    path = _find_tool("mlir-opt")
    if not path:
        raise FileNotFoundError(
            "mlir-opt not found. Build LLVM (third-party/llvm) or add it to PATH."
        )
    return path


def artifacts_dir() -> Path:
    """Return (and create) a directory for notebook-generated artifacts."""

    target = Path(os.environ.get("CINNAMON_NOTEBOOK_ARTIFACTS", _TUTORIAL_DIR / "artifacts"))
    target.mkdir(parents=True, exist_ok=True)
    return target


def run(argv: Sequence[os.PathLike[str] | str], *, check: bool = True) -> str:
    """Run ``argv`` and return stdout as text.

    ``argv`` can contain ``Path`` objects; they are coerced to strings.
    ``check`` mirrors ``subprocess.run``.
    Raises ``CalledProcessError`` when ``check`` is true and the
    command fails. Otherwise it returns the captured stdout.
    """

    normalized: List[str] = [str(Path(arg)) if isinstance(arg, Path) else str(arg) for arg in argv]
    proc = subprocess.run(
        normalized,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc.stdout


# ---------------------------------------------------------------------------
# Portable build helpers (macOS + Ubuntu)
# ---------------------------------------------------------------------------

def _platform_build_flags(min_macos_version: str = "11.0") -> Tuple[List[str], List[str]]:
    """Return (cflags, ldflags) that make clang work on macOS and Ubuntu.

    - On macOS, points clang at the active SDK and sets a minimum OS version.
      Respects SDKROOT/MINVER if already set in the environment.
    - On Linux, returns empty lists.
    """
    sysname = platform.system()
    cflags: List[str] = []
    ldflags: List[str] = []
    if sysname == "Darwin":
        sdk = os.environ.get("SDKROOT")
        if not sdk:
            try:
                sdk = subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()
                os.environ["SDKROOT"] = sdk  # also useful for subprocesses
            except Exception:
                sdk = None
        minver = os.environ.get("MINVER", min_macos_version)
        if sdk:
            cflags += ["-isysroot", sdk, f"-mmacosx-version-min={minver}"]
            ldflags += ["-isysroot", sdk, f"-mmacosx-version-min={minver}"]
        else:
            # Still pass a deployment target even if SDK detection failed.
            cflags += [f"-mmacosx-version-min={minver}"]
            ldflags += [f"-mmacosx-version-min={minver}"]
    # Linux/Ubuntu: nothing special
    return cflags, ldflags


# ---- Build & timing helpers (kept minimal; no removals) ----

def clang_path() -> Path:
    p = _find_tool("clang")
    if not p:
        raise FileNotFoundError("clang not found. Install/point LLVM toolchain to PATH.")
    return p

def llc_path() -> Path:
    p = _find_tool("llc")
    if not p:
        raise FileNotFoundError("llc not found. Install/point LLVM toolchain to PATH.")
    return p


# NOTE: Original simple compile/link functions are preserved, but now automatically
#       add portable, OS-aware flags so they work on macOS and Ubuntu.

def compile_ll_to_obj(ll_file: Path, obj_file: Optional[Path] = None, opt_level: str = "2") -> Path:
    """Compile LLVM IR (.ll) to an object file using clang (portable flags applied)."""
    ll_file = Path(ll_file).resolve()
    if obj_file is None:
        obj_file = ll_file.with_suffix(".o")
    obj_file = Path(obj_file).resolve()
    os_cflags, _ = _platform_build_flags()
    cmd = [clang_path(), f"-O{opt_level}", *os_cflags, "-c", str(ll_file), "-o", str(obj_file)]
    run(cmd)
    return obj_file

def compile_c_to_obj(c_file: Path, obj_file: Optional[Path] = None,
                     opt_level: str = "2", extra_cflags: Sequence[str] = ()) -> Path:
    """Compile a C source file to an object file using clang (portable flags applied)."""
    c_file = Path(c_file).resolve()
    if obj_file is None:
        obj_file = c_file.with_suffix(".o")
    obj_file = Path(obj_file).resolve()
    os_cflags, _ = _platform_build_flags()
    # Include a safe default C standard; callers can override by passing their own via extra_cflags.
    cmd = [clang_path(), f"-O{opt_level}", "-std=c11", *os_cflags, *extra_cflags,
           "-c", str(c_file), "-o", str(obj_file)]
    run(cmd)
    return obj_file

def link_executable(objs: Sequence[Path], exe_path: Path, extra_ldflags: Sequence[str] = ()) -> Path:
    """Link object files into an executable (portable flags applied)."""
    exe_path = Path(exe_path).resolve()
    _, os_ldflags = _platform_build_flags()
    cmd = [clang_path(), *[str(Path(o)) for o in objs], "-o", str(exe_path), *os_ldflags, *extra_ldflags]
    run(cmd)
    return exe_path

def build_executable(llvm_ir_ll: Path, driver_c: Path, exe_path: Path,
                     opt_level: str = "2",
                     extra_cflags: Sequence[str] = (),
                     extra_ldflags: Sequence[str] = ()):
    """Convenience: .ll -> obj, driver.c -> obj, then link -> exe (portable)."""
    exe_path = Path(exe_path).resolve()
    ll_obj = compile_ll_to_obj(llvm_ir_ll, exe_path.with_suffix(".ir.o"), opt_level=opt_level)
    c_obj  = compile_c_to_obj(driver_c, exe_path.with_suffix(".drv.o"),
                              opt_level=opt_level, extra_cflags=extra_cflags)
    return link_executable([ll_obj, c_obj], exe_path, extra_ldflags=extra_ldflags)

def time_executable(exe: Path, args=(), warmup: int = 2, repeat: int = 10, env=None):
    """Run an executable multiple times and return timing stats + last stdout."""
    import time, statistics
    exe = Path(exe).resolve()
    if not exe.exists():
        raise FileNotFoundError(f"Executable not found: {exe}")

    base_env = os.environ.copy()
    if env:
        base_env.update(env)

    def once():
        t0 = time.perf_counter()
        p = subprocess.run([str(exe), *map(str, args)],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, env=base_env, check=True)
        return time.perf_counter() - t0, p.stdout

    # Warmup
    for _ in range(warmup):
        once()

    times = []
    last_out = ""
    for _ in range(repeat):
        dt, out = once()
        times.append(dt)
        last_out = out

    return {
        "runs": repeat,
        "min_s": min(times),
        "mean_s": statistics.mean(times),
        "max_s": max(times),
        "stdev_s": statistics.pstdev(times) if repeat > 1 else 0.0,
        "last_stdout": last_out.strip(),
    }


# ---------------------------------------------------------------------------
# Optional: explicit portable variants (kept in case you want to call them directly)
# ---------------------------------------------------------------------------

def platform_build_flags(min_macos_version: str = "11.0") -> Tuple[List[str], List[str]]:
    """Public wrapper for OS-aware flags (cflags, ldflags)."""
    return _platform_build_flags(min_macos_version)

def compile_ll_to_obj_with_flags(ll_file: Path, obj_file: Optional[Path] = None,
                                 opt_level: str = "2",
                                 extra_cflags: Sequence[str] = ()) -> Path:
    ll_file = Path(ll_file).resolve()
    if obj_file is None:
        obj_file = ll_file.with_suffix(".o")
    obj_file = Path(obj_file).resolve()
    os_cflags, _ = _platform_build_flags()
    cmd = [clang_path(), f"-O{opt_level}", *os_cflags, *extra_cflags, "-c", str(ll_file), "-o", str(obj_file)]
    run(cmd)
    return obj_file

def compile_c_to_obj_with_flags(c_file: Path, obj_file: Optional[Path] = None,
                                opt_level: str = "2", c_standard: str = "c11",
                                extra_cflags: Sequence[str] = ()) -> Path:
    c_file = Path(c_file).resolve()
    if obj_file is None:
        obj_file = c_file.with_suffix(".o")
    obj_file = Path(obj_file).resolve()
    os_cflags, _ = _platform_build_flags()
    cmd = [clang_path(), f"-O{opt_level}", f"-std={c_standard}", *os_cflags, *extra_cflags,
           "-c", str(c_file), "-o", str(obj_file)]
    run(cmd)
    return obj_file

def link_executable_with_flags(objects: Sequence[Path], exe_path: Path,
                               extra_ldflags: Sequence[str] = ()) -> Path:
    exe_path = Path(exe_path).resolve()
    _, os_ldflags = _platform_build_flags()
    cmd = [clang_path(), *[str(Path(o)) for o in objects], "-o", str(exe_path), *os_ldflags, *extra_ldflags]
    run(cmd)
    return exe_path

def build_executable_portable(llvm_ir_ll: Path, driver_c: Path, exe_path: Path,
                              opt_level: str = "2", c_standard: str = "c11",
                              extra_cflags: Sequence[str] = (),
                              extra_ldflags: Sequence[str] = ()):
    exe_path = Path(exe_path).resolve()
    ll_obj = compile_ll_to_obj_with_flags(llvm_ir_ll, exe_path.with_suffix(".ir.o"),
                                          opt_level=opt_level, extra_cflags=extra_cflags)
    c_obj  = compile_c_to_obj_with_flags(driver_c, exe_path.with_suffix(".drv.o"),
                                         opt_level=opt_level, c_standard=c_standard,
                                         extra_cflags=extra_cflags)
    return link_executable_with_flags([ll_obj, c_obj], exe_path, extra_ldflags=extra_ldflags)


__all__ = [
    "artifacts_dir", "mlir_opt_path", "run", "tools",
    "clang_path", "llc_path",
    "compile_ll_to_obj", "compile_c_to_obj", "link_executable",
    "build_executable", "time_executable",
    "mlir_translate",
    "platform_build_flags",
    "compile_ll_to_obj_with_flags",
    "compile_c_to_obj_with_flags",
    "link_executable_with_flags",
    "build_executable_portable",
]
