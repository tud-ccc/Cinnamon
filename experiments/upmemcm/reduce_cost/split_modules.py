#!/usr/bin/env python3
"""Split a split-input-file MLIR file into individual files, one per module.

Usage: python3 split_modules.py <input.mlir> <out_dir> <name0> [<name1> ...]
The first chunk (usually an empty module) is always skipped.
"""
import sys
import pathlib

src, out_dir, *names = sys.argv[1:]
parts = pathlib.Path(src).read_text().split("\n// -----\n")
chunks = parts[1:]  # skip the leading empty module

assert len(chunks) == len(names), (
    f"Expected {len(names)} modules, found {len(chunks)}")

pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
for name, chunk in zip(names, chunks):
    (pathlib.Path(out_dir) / f"{name}.mlir").write_text(chunk)
    print(f"  wrote {out_dir}/{name}.mlir")
