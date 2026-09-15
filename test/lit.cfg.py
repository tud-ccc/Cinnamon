# -*- Python -*-

import os
import sys
import importlib.util

import lit.formats
import lit.util

from lit.llvm import llvm_config


# The tests under Python/ drive the torch backend, so every module it imports
# has to be there. cinnamon on its own does not settle that: it is a pypi
# dependency of the environment, so it is importable wherever pixi has run,
# including jobs that only unpack a build tree. torch-mlir is the one that
# says a build actually happened here -- it is not on PyPI, build-torch.sh
# installs it from the submodule.
def _importable(name):
    try:
        return importlib.util.find_spec(name) is not None
    except ImportError:
        return False


_missing = [m for m in ("cinnamon", "torch", "torch_mlir") if not _importable(m)]
has_cinnamon_module = not _missing
if has_cinnamon_module:
    print("INFO: running the python tests")
else:
    print(f"WARNING: skipping the python tests; no {', '.join(_missing)}")
    print("INFO: using python interpreter:", sys.executable)

# name: The name of this test suite.
config.name = "cinm-mlir"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir", ".test"] + ([".py"] if has_cinnamon_module else [])

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)
# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.base2_obj_root, "test")

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ["Inputs", "CMakeLists.txt", "README.md", "LICENSE", "lit.cfg.py"]

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

# The dialect runtimes the python backend dlopens are built against the C++
# runtime of the interpreter's environment. The torch wheel resolves
# libstdc++ through the default search path and loads the system one into the
# process first, which is older than what those runtimes need.
llvm_config.with_environment(
    "LD_LIBRARY_PATH",
    os.path.join(os.path.dirname(os.path.dirname(config.python_executable)), "lib"),
    append_path=True,
)

llvm_config.use_default_substitutions()

# Tweak the PATH to include the tools dir.
config.base2_tools_dir = os.path.join(config.base2_obj_root, "bin")
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.base2_tools_dir, append_path=True)

tool_dirs = [
    config.base2_tools_dir,
    config.llvm_tools_dir,
    os.path.dirname(config.python_executable),
]
tools = ["cinm-opt", "python"]

llvm_config.add_tool_substitutions(tools, tool_dirs)
