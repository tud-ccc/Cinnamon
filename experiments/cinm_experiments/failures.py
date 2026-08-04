"""Collect and classify the compile/run failures left behind under a prim's
compile_root/run_root, for a triaged crash report instead of scrolling
doit's live stdout -- currently the only place compile_configs/run_configs
failures surface (compile_run.py's per-config "FAIL compile"/"FAIL run"
print statements)."""

from __future__ import annotations

import pathlib
import re

import pandas as pd

from .aggregate import iter_config_dirs

# Ordered (first match wins) regex -> short signature name, checked against
# whatever failure text is available on disk. Add new patterns here as new
# failure modes are discovered instead of letting them all fall into "other".
# Names deliberately don't repeat the "compile"/"run" stage -- that's already
# its own `stage` column (see summarize).
_SIGNATURES = [
    (re.compile(r"[Aa]llocation error"), "dpu_alloc"),
    (re.compile(r"Segmentation fault"), "segfault"),
    (re.compile(r"Assertion `.*' failed"), "assertion"),
    (re.compile(r"PLEASE submit a bug report"), "llvm_crash"),
    (re.compile(r"cinm-opt failed"), "cinm_opt_error"),
    (re.compile(r"make failed|make(?:\[\d+\])?: \*\*\*"), "make_error"),
    (re.compile(r"not compiled"), "missing_binary"),
]


def _classify_and_excerpt(text: str, window: int = 200) -> tuple[str, str]:
    """(signature, excerpt). If a signature regex matched, the excerpt is a
    window of context around that match -- deliberately not just "the last N
    chars", since older logs can have the subprocess's own output and
    cinmopt._run's echoed command line in either order (that write wasn't
    flushed before the fd was handed to the subprocess; fixed now, but
    existing logs on disk may still be affected) -- a fixed offset from
    either end isn't reliably where the interesting text is. Falls back to
    the start of whatever follows the echoed command line when nothing
    matched, since that's where a subprocess's own output normally starts."""
    for pattern, name in _SIGNATURES:
        m = pattern.search(text)
        if m:
            lo = max(0, m.start() - window)
            hi = min(len(text), m.end() + window)
            return name, text[lo:hi].strip()
    if not text:
        return "unknown", ""  # no log file at all, or one never flushed to disk
    # cinmopt._run's log always starts with the echoed command line + a blank
    # line before any subprocess output; if nothing follows, the subprocess
    # exited non-zero without printing anything -- most likely killed by a
    # signal (e.g. OOM) rather than a diagnosed error, worth its own bucket
    # rather than lumping in with "other" (which implies unrecognized *text*).
    _, _, body = text.partition("\n\n")
    body = body.strip()
    if not body:
        return "silent_failure", ""
    return "other", body[: window * 2]


def collect_compile_failures(
    compile_dir: pathlib.Path, *, only_fn: str | None = None
) -> pd.DataFrame:
    """[fn_name, label, ...params, stage, signature, excerpt] for every
    config under compile_dir missing its bin/bench_<fn> -- i.e.
    compile_run.compile_config returned ok=False (compile_run.py:61-112).
    config.csv always exists (written before lowering starts); cinm-opt.log
    and/or make_stderr.txt (whichever a run got as far as writing) supply
    the failure text.

    only_fn restricts the walk to a single fn_name, for cheap per-function
    reporting right after that function's compile step (dodo.py's
    _compile_fn) instead of re-scanning every function's directory."""
    compile_dir = pathlib.Path(compile_dir)
    rows = []
    for fn_name, config_dir in iter_config_dirs(compile_dir):
        if only_fn and fn_name != only_fn:
            continue
        config_csv = config_dir / "config.csv"
        if not config_csv.exists():
            continue
        if (config_dir / "bin" / f"bench_{fn_name}").exists():
            continue

        config_meta = pd.read_csv(config_csv).iloc[0].to_dict()
        make_stderr = config_dir / "make_stderr.txt"
        cinm_opt_log = config_dir / "cinm-opt.log"
        if make_stderr.exists():
            text = make_stderr.read_text(errors="replace")
        elif cinm_opt_log.exists():
            text = cinm_opt_log.read_text(errors="replace")
        else:
            text = ""
        signature, excerpt = _classify_and_excerpt(text)
        rows.append(
            {
                **config_meta,
                "stage": "compile",
                "signature": signature,
                "excerpt": excerpt,
            }
        )
    return pd.DataFrame(rows)


def collect_run_failures(
    run_dir: pathlib.Path, compile_dir: pathlib.Path, *, only_fn: str | None = None
) -> pd.DataFrame:
    """[fn_name, label, ...params, stage, signature, excerpt] for every
    config under run_dir with a missing/empty output/ dir -- i.e.
    compile_run.run_config returned ok=False (compile_run.py:115-136).
    error.txt (written by run_config alongside output/ on failure) supplies
    the failure text. Skips configs that never compiled -- those are
    reported by collect_compile_failures instead, and run_configs never
    even creates a run_dir entry for them (compile_run.py:200-201)."""
    run_dir = pathlib.Path(run_dir)
    compile_dir = pathlib.Path(compile_dir)

    run_dir.parent.mkdir(parents=True, exist_ok=True)
    compile_dir.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for fn_name, config_dir in iter_config_dirs(run_dir):
        if only_fn and fn_name != only_fn:
            continue
        output_dir = config_dir / "output"
        if output_dir.exists() and any(output_dir.iterdir()):
            continue

        fn_compile_dir = compile_dir / fn_name / config_dir.name
        config_csv = fn_compile_dir / "config.csv"
        if (
            not config_csv.exists()
            or not (fn_compile_dir / "bin" / f"bench_{fn_name}").exists()
        ):
            continue  # never compiled -- already reported as a compile failure

        config_meta = pd.read_csv(config_csv).iloc[0].to_dict()
        error_txt = config_dir / "error.txt"
        text = error_txt.read_text(errors="replace") if error_txt.exists() else ""
        signature, excerpt = _classify_and_excerpt(text)
        rows.append(
            {
                **config_meta,
                "stage": "run",
                "signature": signature,
                "excerpt": excerpt,
            }
        )
    return pd.DataFrame(rows)


def collect_failures(
    run_dir: pathlib.Path, compile_dir: pathlib.Path, *, only_fn: str | None = None
) -> pd.DataFrame:
    """Concatenate collect_compile_failures + collect_run_failures into one
    [fn_name, label, ...params, stage, signature, excerpt] frame."""
    frames = [
        f
        for f in (
            collect_compile_failures(compile_dir, only_fn=only_fn),
            collect_run_failures(run_dir, compile_dir, only_fn=only_fn),
        )
        if not f.empty
    ]
    if not frames:
        return pd.DataFrame(
            columns=["fn_name", "label", "stage", "signature", "excerpt"]
        )
    return pd.concat(frames, ignore_index=True)


def summarize(failures: pd.DataFrame) -> str:
    """One-line-per-(stage,signature) failure count, e.g. for printing right
    after a doit compile/bench step."""
    if failures.empty:
        return "  no failures"
    counts = (
        failures.groupby(["stage", "signature"]).size().sort_values(ascending=False)
    )
    return "\n".join(f"  {stage}:{sig}: {n}" for (stage, sig), n in counts.items())
