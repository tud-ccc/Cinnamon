"""The tools the torch backend shells out to, by the names it calls them.

A thin naming layer over cinnamon.paths, which is where an installed build is
actually found. `clang` is the exception: it is only the linker driver for the
compiled module, it reads none of our IR, so the environment's compiler is
right and an installed one would not be.
"""

import os
import shutil

from ... import paths


class ResourcePaths:
    def torch_mlir_opt():
        return str(paths.tool("torch-mlir-opt"))

    def cinm_opt():
        return str(paths.tool("cinm-opt"))

    def mlir_translate():
        return str(paths.tool("mlir-translate"))

    def llc():
        return str(paths.tool("llc"))

    def clang():
        # CC is what the build was configured with, so it matches the C++
        # runtime the installed libraries expect.
        cc = os.environ.get("CC")
        if not cc:
            # Never a bare `clang`: UPMEM_HOME/bin is on PATH whenever the
            # SDK is set up and ships a clang 12 under that name, which is a
            # DPU compiler and cannot link a host object.
            cc = shutil.which("cc")
        if not cc:
            raise paths.CinnamonNotInstalled(
                "No C compiler to link with: set CC, or put cc on PATH."
            )
        upmem_home = os.environ.get("UPMEM_HOME")
        if upmem_home and os.path.realpath(cc).startswith(
            os.path.realpath(upmem_home) + os.sep
        ):
            raise paths.CinnamonNotInstalled(
                f"'{cc}' is the UPMEM SDK's compiler, which targets DPUs. "
                "Set CC to a host compiler."
            )
        return cc

    def memristor_runtime():
        return str(paths.memristor_runtime())
