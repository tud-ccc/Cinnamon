# The image the DSE tutorial runs in (github.com/tud-ccc/esweek2026-tutorial):
# Cinnamon built against the prebuilt LLVM, inside the pixi environment that
# also holds Jupyter. No Torch-MLIR, no gem5: the tutorial needs cinm-opt and
# Python.
#
# Build from the repository root, naming the LLVM revision the submodule pins
# (the context has no .git to read it from):
#
#   docker/tutorial/build.sh            # does both
#
# Layers are ordered by how often they change: the pixi environment on the lock
# file, the conan packages on the conan inputs, Cinnamon itself on everything.
FROM ubuntu:24.04

ARG PIXI_VERSION=v0.78.0
ENV DEBIAN_FRONTEND=noninteractive \
    PIXI_HOME=/opt/pixi \
    PATH=/opt/pixi/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git \
 && rm -rf /var/lib/apt/lists/* \
 && curl -fsSL https://pixi.sh/install.sh | PIXI_VERSION=${PIXI_VERSION} bash

WORKDIR /opt/cinnamon

# 1. The environment: compiler, cmake, ninja, Python, Jupyter, conan, ... The
#    two in-tree Python projects are path dependencies of the manifest, so the
#    solver needs them here already; they are small and rarely change.
COPY pixi.toml pixi.lock ./
COPY python/cinnamon python/cinnamon
COPY python/experiments python/experiments
RUN pixi install --locked -e default \
 && pixi clean cache --yes

# 2. The conan packages, built from source with the environment's compiler.
#    Only what decides which packages are built is copied in, so this layer
#    survives ordinary commits.
COPY conanfile.txt ./
COPY third-party/conan-recipes third-party/conan-recipes
COPY .github/workflows/common.sh .github/workflows/build-cinnamon.sh .github/workflows/
RUN pixi run -e default conan-recipes \
 && pixi run -e default .github/workflows/build-cinnamon.sh -conan-only -no-python-venv -no-upmem \
 && pixi run -e default conan cache clean "*" --source --build --temp

# 3. Cinnamon, against the prebuilt LLVM of the pinned revision: build-llvm.sh
#    downloads the fork's release for it rather than building anything. Object
#    files and the ccache are dead weight once the binaries exist.
ARG LLVM_REVISION
RUN test -n "$LLVM_REVISION" || { echo "LLVM_REVISION build-arg is required (git ls-tree --object-only HEAD third-party/llvm)" >&2; exit 1; }
COPY . .
RUN pixi run -e default bash -c ' \
      set -e; \
      LLVM_PREBUILT=always LLVM_REVISION='"$LLVM_REVISION"' .github/workflows/build-llvm.sh -no-python-venv; \
      .github/workflows/build-cinnamon.sh -no-python-venv -no-upmem' \
 && find build -name '*.o' -delete \
 && rm -rf .ccache \
 && pixi run -e default conan cache clean "*" --source --build --temp

# 4. What Binder expects: a user jovyan with uid 1000, and the tools on PATH
#    for every process it starts (it does not source profiles). The ubuntu
#    image ships a user "ubuntu" at that uid; it is replaced.
ARG NB_USER=jovyan
ARG NB_UID=1000
RUN if getent passwd ${NB_UID} >/dev/null; then userdel -r "$(getent passwd ${NB_UID} | cut -d: -f1)"; fi \
 && useradd --create-home --uid ${NB_UID} --shell /bin/bash ${NB_USER}
ENV PATH=/opt/cinnamon/.pixi/envs/default/bin:/opt/cinnamon/build/bin:$PATH \
    HOME=/home/${NB_USER}
USER ${NB_USER}
WORKDIR /home/${NB_USER}
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser"]
