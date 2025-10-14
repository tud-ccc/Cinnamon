# -*- Python -*-

import os
import sys
import importlib.util

import lit.formats
import lit.util

from lit.llvm import llvm_config

has_cinnamon_module = importlib.util.find_spec("cinnamon") is not None
if has_cinnamon_module:
    print("INFO: cinnamon module found; running python tests")
else:
    print("WARNING: cinnamon module not found; skipping python tests")
    print("INFO: using python interpreter:", sys.executable)

# name: The name of this test suite.
config.name = "cinm-mlir"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir"] + ([".py"] if has_cinnamon_module else [])

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
