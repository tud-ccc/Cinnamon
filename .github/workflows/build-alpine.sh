#!/bin/bash
set -euo pipefail

script_dir="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
# shellcheck source=/dev/null
source "$script_dir/common.sh"

alpine_repo_url="${ALPINE_REPO_URL:-https://github.com/gem5-X/ALPINE.git}"
alpine_revision="${ALPINE_REV:-master}"
alpine_src_dir="$project_root/third-party/ALPINE"
alpine_docker_dir="$project_root/third-party/alpine"
docker_image_tag="${ALPINE_DOCKER_TAG:-alpine-gem5:latest}"

rebuild_docker=0
build_in_container=1

for arg in "$@"; do
  case "$arg" in
    -rebuild-docker) rebuild_docker=1 ;;
    -no-container-build) build_in_container=0 ;;
  esac
done

status "Preparing ALPINE checkout in: $alpine_src_dir"
if [[ ! -d "$alpine_src_dir" ]]; then
  git_clone_revision "$alpine_repo_url" "$alpine_revision" "$alpine_src_dir"
else
  info "ALPINE already present; skipping clone (set ALPINE_REV or delete dir to re-clone)"
fi

util_py="$alpine_src_dir/gem5-X-ALPINE/src/python/m5/util/__init__.py"
if [[ -f "$util_py" ]]; then
  pybin="$(command -v python3 || command -v python || true)"
  if [[ -n "$pybin" ]]; then
    "$pybin" - "$util_py" <<'PY'
import io
import re
import sys

path = sys.argv[1]
original = io.open(path, 'r', encoding='utf-8').read()
patched = original.replace(
    "return map(lambda x: int(re.match('\\d+', x).group()), v.split('.'))",
    "return [int(re.match('\\d+', x).group(0)) for x in v.split('.') if re.match('\\d+', x)]"
)

if original != patched:
    io.open(path, 'w', encoding='utf-8').write(patched)
    print('Patched version parser in', path)
else:
    print('Version parser already patched in', path)
PY
  else
    warning "Skipping util.py patch; no python interpreter found"
  fi
else
  info "No util.py patch needed (file not present)"
fi

if [[ "$build_in_container" -eq 0 ]]; then
  info "Container build disabled (-no-container-build); exiting after checkout"
  exit 0
fi

command -v docker >/dev/null 2>&1 || {
  error "Docker is required to build ALPINE gem5; install Docker or rerun with -no-container-build"
  exit 1
}

mkdir -p "$alpine_docker_dir"
dockerfile_path="$alpine_docker_dir/Dockerfile"

cat <<'EOF' >"$dockerfile_path"
# ===== gem5 ready image =====
FROM ubuntu:20.04

SHELL ["/bin/bash","-c"]
ENV DEBIAN_FRONTEND=noninteractive TZ=UTC LANG=C.UTF-8

# --- System packages (compilers, libs, qemu, dtc, etc.) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git wget curl ca-certificates pkg-config \
    m4 zlib1g zlib1g-dev \
    libprotobuf-dev protobuf-compiler libprotoc-dev \
    libgoogle-perftools-dev libboost-all-dev \
    device-tree-compiler \
    gcc-arm-linux-gnueabihf gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    libc6-dev-arm64-cross libstdc++-10-dev-arm64-cross \
    qemu-system qemu-user qemu-user-static qemu-utils binfmt-support \
    autoconf automake libtool bison flex libpcre3 libpcre3-dev \
    diod \
    python2 libpython2.7 libpython2.7-dev \
    && rm -rf /var/lib/apt/lists/*

# --- SWIG 3.0.8 (needed by some older gem5-X flows) ---
RUN wget -q https://downloads.sourceforge.net/swig/swig-3.0.8.tar.gz \
    && tar -xzf swig-3.0.8.tar.gz && cd swig-3.0.8 \
    && ./configure --prefix=/usr/local \
    && make -j"$(nproc)" && make install \
    && cd / && rm -rf swig-3.0.8 swig-3.0.8.tar.gz

# --- Miniconda (so we can have both py3 and py2 envs) ---
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3 \
    && rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/opt/miniconda3/bin:$PATH

# Accept Anaconda ToS to avoid non-interactive build failures
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && conda config --set always_yes yes --set changeps1 no

# --- Python 3 env for *upstream gem5* ---
RUN conda create -n py3 python=3.10 pip \
    && conda run -n py3 pip install "scons>=4.5" pydot

# --- Python 2.7 env for *legacy gem5-X* (PDF) ---
RUN conda create -n py27 python=2.7 pip \
    && conda run -n py27 pip install "scons==3.0.0"

# Auto-activate py3 in interactive shells
RUN echo 'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate py3' >> /etc/bash.bashrc

WORKDIR /project
EOF

if [[ "$rebuild_docker" -eq 1 ]]; then
  status "Rebuilding Docker image $docker_image_tag"
  docker build -t "$docker_image_tag" "$alpine_docker_dir"
elif ! docker image inspect "$docker_image_tag" >/dev/null 2>&1; then
  status "Docker image $docker_image_tag not found; building it now"
  docker build -t "$docker_image_tag" "$alpine_docker_dir"
else
  info "Using existing Docker image $docker_image_tag"
fi

status "Building gem5-X (ARM opt) inside Docker container"
docker run --rm \
  -u "$(id -u)":"$(id -g)" \
  -v "$alpine_src_dir":/project/ALPINE \
  -w /project/ALPINE/gem5-X-ALPINE \
  "$docker_image_tag" \
  bash -lc 'set -euo pipefail; source /opt/miniconda3/etc/profile.d/conda.sh; conda activate py3; scons build/ARM/gem5.opt -j"$(nproc)"'

status "gem5 binary available at: $alpine_src_dir/gem5-X-ALPINE/build/ARM/gem5.opt"
