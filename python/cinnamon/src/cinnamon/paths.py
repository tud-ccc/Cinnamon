"""Where the things Cinnamon builds are.

Everything `just install` puts into an environment lives under one directory,
and this module is the only thing that knows which. Callers ask for a tool or
a suite by name; they never join a path onto a checkout root.

The directory is `libexec/cinnamon` of the active environment -- a subtree of
its own, because our LLVM is a whole toolchain and the environment is shared
with conda packages carrying an LLVM of theirs. `CINNAMON_PREFIX` overrides
it, which is how a pipeline runs against a build other than the one in the
environment it was started from.
"""

import os
import pathlib
import sys

# Relative to an environment root. Matches the justfile's `install_dir`.
_SUBDIR = pathlib.PurePath("libexec", "cinnamon")

# Under the install directory. The tools and the libraries they load keep the
# usual layout; the rest is where `cmake --install` puts it.
_BENCHMARKS = pathlib.PurePath("share", "cinnamon", "benchmarks")
_REFERENCE_MODEL = pathlib.PurePath("share", "cinnamon", "cnm-cost-model", "Predictor")
_UPMEM_RUNTIME_INCLUDE = pathlib.PurePath("include", "cinm-mlir", "runtime", "Upmem")


class CinnamonNotInstalled(RuntimeError):
    """Raised when no installed Cinnamon can be found, or one is incomplete."""


def _looks_installed(d: pathlib.Path) -> bool:
    """Whether `d` is a Cinnamon install directory. cinm-opt stands for the
    whole install: it is what `cmake --install` puts there first and what
    every caller needs."""
    return (d / "bin" / "cinm-opt").is_file()


def _candidates() -> list[pathlib.Path]:
    """The directories to look in, in order, with why each is a candidate."""
    override = os.environ.get("CINNAMON_PREFIX")
    if override:
        # Both spellings, because either is a reasonable thing to have in
        # hand: the install directory itself, or the environment holding it.
        root = pathlib.Path(override)
        return [root, root / _SUBDIR]
    # The environment this interpreter is running in. sys.prefix rather than
    # a path derived from this file, so that it is the same answer whether the
    # package was installed editable from a checkout or as a wheel.
    return [pathlib.Path(sys.prefix) / _SUBDIR]


def install_dir() -> pathlib.Path:
    """The directory Cinnamon is installed in.

    Raises CinnamonNotInstalled if there is none, with the two ways out: build
    and install one, or point at an existing one.
    """
    candidates = _candidates()
    for d in candidates:
        if _looks_installed(d):
            return d
    looked = "\n".join(f"  {d}" for d in candidates)
    raise CinnamonNotInstalled(
        "No installed Cinnamon found. Looked in:\n"
        f"{looked}\n"
        "Run `just install` in a cinm-mlir checkout to create one, or set "
        "CINNAMON_PREFIX to an environment that already has one."
    )


def bin_dir() -> pathlib.Path:
    return install_dir() / "bin"


def lib_dir() -> pathlib.Path:
    return install_dir() / "lib"


def tool(name: str) -> pathlib.Path:
    """The path of an installed executable.

    Covers our own tools (cinm-opt, cinm-translate, cinm-compile-dpu) and the
    LLVM ones installed beside them (opt, llc, mlir-translate). Those come
    from the LLVM this was built against, on purpose: IR one of our tools
    emits is not necessarily readable by another LLVM's.
    """
    path = bin_dir() / name
    if not path.is_file():
        raise CinnamonNotInstalled(
            f"'{name}' is not in {bin_dir()}. The install there is incomplete "
            "or was made by a different build."
        )
    return path


def benchmarks_dir() -> pathlib.Path:
    """The benchmark suites: prim/, multiop/ and cinm1/, each holding a
    <name>.mlir beside the <name>.cpp driver that runs it. Installed with the
    compiler that lowers them, because a suite and that compiler are one
    version of one thing."""
    path = install_dir() / _BENCHMARKS
    if not path.is_dir():
        raise CinnamonNotInstalled(f"No benchmark suites in {path}.")
    return path


def reference_model_dir() -> pathlib.Path:
    """The analytical cost model the evaluation prices programs against. Run
    as a subprocess from this directory, which is why it is a directory rather
    than the one script: it reads its kernel tables relative to itself."""
    path = install_dir() / _REFERENCE_MODEL
    if not path.is_dir():
        raise CinnamonNotInstalled(f"No reference model in {path}.")
    return path


def upmem_runtime_include() -> pathlib.Path:
    """Headers the bench drivers compile against (upmem_rt.h, timers.h)."""
    return install_dir() / _UPMEM_RUNTIME_INCLUDE


def upmem_runtime_lib() -> pathlib.Path:
    """The static UPMEM runtime the bench drivers link. Only present when the
    build had the UPMEM SDK; there is nothing to run on a machine without one
    anyway."""
    return lib_dir() / "libUpmemDialectRuntime.a"


def memristor_runtime() -> pathlib.Path:
    """The shared memristor runtime, which the torch backend dlopens."""
    return lib_dir() / "libMemristorDialectRuntime.so"
