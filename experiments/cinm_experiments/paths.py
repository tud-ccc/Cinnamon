import pathlib
import sys
import os

EXPERIMENTS_DIR = pathlib.Path(__file__).resolve().parent.parent
ROOT = EXPERIMENTS_DIR.parent
DEFAULT_CINM_OPT = ROOT / "build" / "bin" / "cinm-opt"
DEFAULT_CINM_OPT = pathlib.Path(os.environ.get("CINM_OPT", DEFAULT_CINM_OPT))
COMPILE_MAKEFILE_DIR = EXPERIMENTS_DIR / "cinm_experiments"


def python_bin() -> str:
    """The experiments venv's python if it exists, else the current interpreter."""
    venv = ROOT / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable
