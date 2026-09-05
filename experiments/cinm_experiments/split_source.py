"""Splitting a `// -----`-separated multi-function MLIR module into one file
per function, so individual functions can be driven separately (needed to
pin dpus/tasklets per function)."""

from __future__ import annotations

import pathlib
import re

_FN_RE = re.compile(r"func\.func @(\w+)")

#: The chunk separator. This module splits on it anchored to its own line,
#: but MLIR's --split-input-file splits on it as a bare substring, so a line
#: like `// // -----` -- how a prim source comments out a whole function --
#: stays inside a chunk here and then splits it again downstream. The extra
#: empty modules that produces travel all the way to mlir-translate, where
#: they surface as host ops that were never converted. Chunks are written
#: with the marker defused so both splitters agree that a chunk is one module.
_MARKER = "// -----"
_DEFUSED_MARKER = "// - - - - -"


def list_functions(src_mlir: pathlib.Path) -> list[str]:
    """Function names in split-chunk order, without writing anything --
    lets callers (e.g. doit task generators) declare split_source()'s output
    paths ahead of time, before actually running the split."""
    text = pathlib.Path(src_mlir).read_text()
    chunks = text.split("\n// -----\n")[1:]
    names = []
    for chunk in chunks:
        m = _FN_RE.search(chunk)
        if not m:
            raise ValueError(f"{src_mlir}: no func.func found in a split chunk")
        names.append(m.group(1))
    return names


def split_source(
    src_mlir: pathlib.Path, out_dir: pathlib.Path
) -> dict[str, pathlib.Path]:
    """Split a `// -----`-separated multi-function module into one .mlir file
    per function. Returns {fn_name: path}.

    Each chunk (skipping the leading one, normally an empty module) is named
    after the first `func.func @name` found within it."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = pathlib.Path(src_mlir).read_text()
    chunks = text.split("\n// -----\n")[1:]
    modules: dict[str, pathlib.Path] = {}
    for chunk in chunks:
        m = _FN_RE.search(chunk)
        if not m:
            raise ValueError(f"{src_mlir}: no func.func found in a split chunk")
        name = m.group(1)
        path = out_dir / f"{name}.mlir"
        path.write_text(chunk.replace(_MARKER, _DEFUSED_MARKER))
        modules[name] = path
    return modules
