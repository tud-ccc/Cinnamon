# syntax=docker/dockerfile:1.4
#
# Cinnamon ESWEEK tutorial image: compiler (LLVM, Torch-MLIR, UPMEM SDK, Cinnamon)
# + simulator (ALPINE gem5-X) + Jupyter, Binder-compatible (UID 1000, repo in $HOME).
#
# Build:  docker build --build-arg JOBS=8 -t cinnamon-tutorial .
# Run:    docker run -it --rm -p 8888:8888 cinnamon-tutorial \
#           jupyter lab --ip=0.0.0.0 --port=8888

############################################################################
# Stage 1: ALPINE gem5-X  (replaces the docker-in-docker step of build-alpine.sh)
############################################################################
FROM ubuntu:20.04 AS alpine
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive TZ=UTC LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential git wget curl ca-certificates pkg-config m4 \
    zlib1g zlib1g-dev \
    libprotobuf-dev protobuf-compiler libprotoc-dev \
    libgoogle-perftools-dev libboost-all-dev \
    device-tree-compiler \
    gcc-arm-linux-gnueabihf gcc-aarch64-linux-gnu g++-aarch64-linux-gnu \
    libc6-dev-arm64-cross libstdc++-10-dev-arm64-cross \
    qemu-system qemu-user qemu-user-static qemu-utils binfmt-support \
    autoconf automake libtool bison flex libpcre3 libpcre3-dev \
    diod \
    python2 libpython2.7 libpython2.7-dev \
    python3 python3-pip \
    gcc-7 g++-7 gcc-8 g++-8 \
 && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 90 \
 && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 90 \
 && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 80 \
 && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 80

RUN wget -q https://downloads.sourceforge.net/swig/swig-3.0.8.tar.gz \
 && tar -xzf swig-3.0.8.tar.gz && cd swig-3.0.8 \
 && ./configure --prefix=/usr/local && make -j"$(nproc)" && make install \
 && cd / && rm -rf swig-3.0.8 swig-3.0.8.tar.gz

RUN pip3 install --no-cache-dir "scons>=4.5" pydot

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3 \
 && rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/opt/miniconda3/bin:$PATH
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
 && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
 && conda config --set always_yes yes --set changeps1 no \
 && conda create -n py27 python=2.7 pip \
 && conda run -n py27 pip install "scons==3.0.0"
ENV PATH=/opt/miniconda3/envs/py27/bin:$PATH

# Pin to a commit for reproducibility (default mirrors build-alpine.sh)
ARG ALPINE_REV=master
RUN git clone https://github.com/gem5-X/ALPINE.git /src/ALPINE \
 && cd /src/ALPINE && git checkout "${ALPINE_REV}"

# Same version-parser patch as build-alpine.sh
RUN python3 - /src/ALPINE/gem5-X-ALPINE/src/python/m5/util/__init__.py <<'PY'
import io, os, sys
p = sys.argv[1]
if os.path.isfile(p):
    s = io.open(p, encoding='utf-8').read()
    t = s.replace(
        "return map(lambda x: int(re.match('\\d+', x).group()), v.split('.'))",
        "return [int(re.match('\\d+', x).group(0)) for x in v.split('.') if re.match('\\d+', x)]")
    if s != t:
        io.open(p, 'w', encoding='utf-8').write(t)
PY

# Compiler wrappers that drop -Werror (same as build-alpine.sh)
RUN mkdir -p /opt/wrap && cat > /opt/wrap/strip <<'EOF'
#!/usr/bin/env bash
real="$1"; shift; args=()
for a in "$@"; do
  [[ "$a" == -Werror || "$a" == -pedantic-errors || "$a" == -Werror=* ]] && continue
  args+=("$a")
done
exec "$real" "${args[@]}"
EOF
RUN printf '#!/usr/bin/env bash\nexec /opt/wrap/strip /usr/bin/gcc-7 "$@"\n' > /opt/wrap/gcc \
 && printf '#!/usr/bin/env bash\nexec /opt/wrap/strip /usr/bin/g++-7 "$@"\n' > /opt/wrap/g++ \
 && chmod +x /opt/wrap/*

ARG JOBS=8
RUN cd /src/ALPINE/gem5-X-ALPINE \
 && CC=/opt/wrap/gcc CXX=/opt/wrap/g++ scons build/ARM/gem5.opt -j "${JOBS}"

# Export: sources/configs without build intermediates, the binary, and its
# non-glibc shared libraries (protobuf, tcmalloc, ... from Ubuntu 20.04)
RUN mkdir -p /export/lib \
 && tar -C /src --exclude='ALPINE/gem5-X-ALPINE/build' -cf - ALPINE | tar -C /export -xf - \
 && mkdir -p /export/ALPINE/gem5-X-ALPINE/build/ARM \
 && cp /src/ALPINE/gem5-X-ALPINE/build/ARM/gem5.opt /export/ALPINE/gem5-X-ALPINE/build/ARM/ \
 && ldd /src/ALPINE/gem5-X-ALPINE/build/ARM/gem5.opt \
    | awk '/=> \//{print $3}' \
    | grep -vE '/(libc|libm|libdl|libpthread|librt|ld-linux)[.-]' \
    | xargs -r -I{} cp -L {} /export/lib/

############################################################################
# Stage 2: Cinnamon toolchain + Jupyter (final, Binder-compatible)
############################################################################
FROM ubuntu:24.04
SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    CC=clang \
    CXX=clang++ \
    LDFLAGS=-fuse-ld=mold \
    CMAKE_GENERATOR=Ninja

# docker.io removed: not needed (ALPINE comes from stage 1) and unusable on Binder
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential clang cmake ninja-build mold git wget curl ca-certificates \
    pkg-config libvulkan-dev \
    python3.12 python3.12-dev python3.12-venv python3-pip \
 && rm -rf /var/lib/apt/lists/*

# Binder: UID 1000 (ubuntu:24.04 already has user 'ubuntu' with that UID)
ARG NB_USER=jovyan
ARG NB_UID=1000
RUN userdel -r ubuntu && useradd -m -s /bin/bash -u ${NB_UID} ${NB_USER}
ENV USER=${NB_USER} HOME=/home/${NB_USER}

# Simulator from stage 1: conda py27 at the same path (gem5 links its libpython2.7)
COPY --from=alpine /opt/miniconda3/envs/py27 /opt/miniconda3/envs/py27
COPY --from=alpine /export/lib /opt/alpine/lib
COPY --from=alpine --chown=${NB_UID} /export/ALPINE ${HOME}/third-party/ALPINE
RUN printf '#!/usr/bin/env bash\nexport LD_LIBRARY_PATH=/opt/alpine/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\nexec %s "$@"\n' \
      "${HOME}/third-party/ALPINE/gem5-X-ALPINE/build/ARM/gem5.opt" > /usr/local/bin/gem5-alpine \
 && chmod +x /usr/local/bin/gem5-alpine

# Repository contents into $HOME, owned by the notebook user
COPY --chown=${NB_UID} . ${HOME}
USER ${NB_USER}
WORKDIR ${HOME}

# Compiler: same order as build.sh, minus the ALPINE step. One RUN per step
# so Docker caches LLVM when later steps fail.
RUN .github/workflows/setup-venv.sh
ENV VIRTUAL_ENV=${HOME}/.venv PATH=${HOME}/.venv/bin:$PATH
RUN .github/workflows/build-llvm.sh
RUN .github/workflows/build-torch.sh
RUN .github/workflows/build-upmem.sh
RUN .github/workflows/build-cinnamon.sh

# Jupyter (required by Binder)
RUN pip install --no-cache-dir notebook jupyterlab
