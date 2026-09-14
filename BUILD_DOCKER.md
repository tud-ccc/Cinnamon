# Running the tutorial

The notebooks in `tutorial/notebooks` lower PyTorch models through Cinnamon and
run the result on the ALPINE analog-in-memory model. Everything except gem5
comes from the pixi environment; gem5 needs Docker.

## Cinnamon

```bash
pixi run configure   # prebuilt LLVM, Torch-MLIR and Cinnamon; see BUILDING.md
pixi run notebook    # Jupyter, rooted at tutorial/notebooks
```

Notebooks 00 to 03, 05, 06, 08 and 10 need nothing further: they run `cinm-opt`
and the LLVM tools from `build/bin`, and compile for the host.

Generated IR, objects and binaries land in `tutorial/artifacts`, which is not
tracked. `pixi run clean-notebooks` strips outputs before committing.

## gem5

Notebooks 04, 07, 09 and 11 cross-compile for aarch64 and run the result under
gem5-X-ALPINE.

```bash
.github/workflows/build-alpine.sh
```

This checks out the `third-party/ALPINE` submodule, builds the `alpine-gem5`
Docker image and then `gem5.opt` inside it. Set aside an hour for the first
run: the image carries Ubuntu 20.04, gcc-7, SWIG 3.0.8 built from source and a
Python 2.7 for scons, because gem5-X-ALPINE predates gem5 v20.0. Both the image
and the binary are reused afterwards. `-rebuild-docker` forces the image to be
rebuilt, `ALPINE_CLEAN=1` makes scons start over.

The notebooks reach it through two scripts, which are also usable directly:

```bash
scripts/docker_llvm_compile.sh tutorial/assets/drivers/snn_driver.c kernel.ll
scripts/docker_run.sh tutorial/artifacts/bin/kernel.out
```

The first cross-compiles the IR on the host and links it against the ALPINE
runtime in `runtime/Alpine` inside the container; the second runs gem5 on the
result and writes its output to `out/program.out` beside the binary. Both mount
the repository at `/workspace`, so every path they are given has to be inside
it.

On Apple silicon the image runs under emulation: it is built and run as
`linux/amd64`, which `DOCKER_PLATFORM` overrides.
