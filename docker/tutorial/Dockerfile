# The image the DSE tutorial runs in (github.com/tud-ccc/esweek2026-tutorial):
# cinm-opt built against the prebuilt LLVM, and a Python environment with
# Jupyter. Nothing else -- no Torch-MLIR, no gem5, and no compiler: the build
# happens in a first stage, and only what runs at the tutorial is carried into
# the second.
#
# Build from the repository root, naming the LLVM revision the submodule pins
# (the context has no .git to read it from):
#
#   docker/tutorial/build.sh            # does both
#
# In the build stage, layers are ordered by how often they change: the pixi
# environment on the lock file, the conan packages on the conan inputs,
# Cinnamon itself on everything.

# ─── Stage 1: build ──────────────────────────────────────────────────────────
FROM ubuntu:24.04 AS build

ARG PIXI_VERSION=v0.78.0
ENV DEBIAN_FRONTEND=noninteractive \
    PIXI_HOME=/opt/pixi \
    PATH=/opt/pixi/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git \
 && rm -rf /var/lib/apt/lists/* \
 && curl -fsSL https://pixi.sh/install.sh | PIXI_VERSION=${PIXI_VERSION} bash

WORKDIR /opt/cinnamon
# Release rather than the scripts' RelWithDebInfo default: debug info is most
# of a 130 MB cinm-opt, and nobody attaches a debugger in a tutorial. Set here
# for the whole stage because conan keys its packages on the build type too --
# the conan layer and the Cinnamon build have to agree, or the latter rebuilds
# every package from source a second time.
ENV CMAKE_BUILD_TYPE=Release

# 1. The environment: compiler, cmake, ninja, Python, conan, ... The two
#    in-tree Python projects are path dependencies of the manifest, so the
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
 && pixi run -e default .github/workflows/build-cinnamon.sh -conan-only -no-python-venv -no-upmem

# 3. Cinnamon, against the prebuilt LLVM of the pinned revision: build-llvm.sh
#    downloads the fork's release for it rather than building anything.
ARG LLVM_REVISION
RUN test -n "$LLVM_REVISION" || { echo "LLVM_REVISION build-arg is required (git ls-tree --object-only HEAD third-party/llvm)" >&2; exit 1; }
COPY . .
RUN pixi run -e default bash -c ' \
      set -e; \
      LLVM_PREBUILT=always LLVM_REVISION='"$LLVM_REVISION"' .github/workflows/build-llvm.sh -no-python-venv; \
      .github/workflows/build-cinnamon.sh -no-python-venv -no-upmem'

# 4. What the next stage takes, laid out under /runtime at the paths the
#    binaries have baked in as RUNPATHs:
#    - cinm-opt and cinm-translate with their libraries, and the links to the
#      LLVM tools that build-cinnamon.sh puts next to them; not the unit tests
#      or the language server
#    - the LLVM's libraries and tools; not its headers or Python bindings
#    - the C++ runtime cinm-opt was built against, which is the toolchain's,
#      newer than the system's -- the only two files it needs from the
#      environment (`ldd` says so; conan's libraries are linked statically)
RUN set -e; r=/runtime/opt/cinnamon; \
    mkdir -p $r/build/bin $r/third-party/llvm-prebuilt $r/.pixi/envs/default/lib; \
    cp -a build/lib $r/build/; \
    cp -a build/bin/cinm-opt build/bin/cinm-translate $r/build/bin/; \
    find build/bin -maxdepth 1 -type l -exec cp -a {} $r/build/bin/ \; ; \
    cp -a third-party/llvm-prebuilt/bin third-party/llvm-prebuilt/lib third-party/llvm-prebuilt/cinnamon-llvm-* $r/third-party/llvm-prebuilt/; \
    rm -rf $r/third-party/llvm-prebuilt/lib/cmake $r/third-party/llvm-prebuilt/lib/python*; \
    cp -L .pixi/envs/default/lib/libstdc++.so.6 .pixi/envs/default/lib/libgcc_s.so.1 $r/.pixi/envs/default/lib/; \
    du -sh $r/build $r/third-party/llvm-prebuilt

# ─── Stage 2: what runs ──────────────────────────────────────────────────────
FROM ubuntu:24.04

ARG PIXI_VERSION=v0.78.0
ENV DEBIAN_FRONTEND=noninteractive \
    PIXI_HOME=/opt/pixi \
    PATH=/opt/pixi/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl \
 && rm -rf /var/lib/apt/lists/* \
 && curl -fsSL https://pixi.sh/install.sh | PIXI_VERSION=${PIXI_VERSION} bash

WORKDIR /opt/cinnamon

# The notebook environment: Python, Jupyter and what the tutorial's own package
# imports. Solved from the same lock file as the build; the path dependencies
# are there for the lock file's sake only.
COPY pixi.toml pixi.lock ./
COPY python/cinnamon python/cinnamon
COPY python/experiments python/experiments
RUN pixi install --locked -e notebook \
 && pixi clean cache --yes

COPY --from=build /runtime/ /

# What Binder expects: a user jovyan with uid 1000, and the tools on PATH for
# every process it starts (it does not source profiles). The ubuntu image
# ships a user "ubuntu" at that uid; it is replaced.
ARG NB_USER=jovyan
ARG NB_UID=1000
RUN if getent passwd ${NB_UID} >/dev/null; then userdel -r "$(getent passwd ${NB_UID} | cut -d: -f1)"; fi \
 && useradd --create-home --uid ${NB_UID} --shell /bin/bash ${NB_USER}
ENV PATH=/opt/cinnamon/.pixi/envs/notebook/bin:/opt/cinnamon/build/bin:$PATH \
    HOME=/home/${NB_USER}
USER ${NB_USER}
WORKDIR /home/${NB_USER}
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser"]
