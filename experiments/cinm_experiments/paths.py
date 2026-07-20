import pathlib
import sys

EXPERIMENTS_DIR = pathlib.Path(__file__).resolve().parent.parent
ROOT = EXPERIMENTS_DIR.parent
DEFAULT_CINM_OPT = ROOT / "build" / "bin" / "cinm-opt"
REDUCE_COST_MAKEFILE_DIR = EXPERIMENTS_DIR / "upmemcm" / "reduce_cost"


def python_bin() -> str:
    """The experiments venv's python if it exists, else the current interpreter."""
    venv = ROOT / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable
