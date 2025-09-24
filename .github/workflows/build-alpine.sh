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

if [[ -z "${DOCKER_PLATFORM:-}" ]]; then
  if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    DOCKER_PLATFORM="linux/amd64"
  fi
fi
docker_platform_arg=""
[[ -n "${DOCKER_PLATFORM:-}" ]] && docker_platform_arg="--platform ${DOCKER_PLATFORM}"

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
import io, re, sys
p=sys.argv[1]
s=io.open(p,'r',encoding='utf-8').read()
t=s.replace(
    "return map(lambda x: int(re.match('\\d+', x).group()), v.split('.'))",
    "return [int(re.match('\\d+', x).group(0)) for x in v.split('.') if re.match('\\d+', x)]"
)
if s!=t:
    io.open(p,'w',encoding='utf-8').write(t); print('Patched version parser in', p)
else:
    print('Version parser already patched in', p)
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

cat >"$dockerfile_path" <<'EOF_DOCKER'
FROM ubuntu:20.04
SHELL ["/bin/bash","-c"]
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
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 90 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 90 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 80 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 80 && \
    gcc --version && g++ --version
RUN wget -q https://downloads.sourceforge.net/swig/swig-3.0.8.tar.gz \
 && tar -xzf swig-3.0.8.tar.gz && cd swig-3.0.8 \
 && ./configure --prefix=/usr/local \
 && make -j"$(nproc)" && make install \
 && cd / && rm -rf swig-3.0.8 swig-3.0.8.tar.gz
RUN pip3 install --no-cache-dir "scons>=4.5" pydot
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3 \
 && rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/opt/miniconda3/bin:$PATH
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda config --set always_yes yes --set changeps1 no && \
    conda create -n py27 python=2.7 pip && \
    conda run -n py27 pip install "scons==3.0.0"
LABEL alpine.gem5.gcc7="true"
WORKDIR /project
EOF_DOCKER

need_build=0
if ! docker image inspect "$docker_image_tag" >/dev/null 2>&1; then
  need_build=1
else
  if [[ "$(docker image inspect -f '{{ index .Config.Labels "alpine.gem5.gcc7"}}' "$docker_image_tag" 2>/dev/null || echo)" != "true" ]]; then
    need_build=1
  fi
fi
if [[ "$rebuild_docker" -eq 1 ]]; then need_build=1; fi

if [[ $need_build -eq 1 ]]; then
  status "Building Docker image $docker_image_tag (gcc-7 default; py27 scons baked in)"
  docker build $docker_platform_arg --pull -t "$docker_image_tag" "$alpine_docker_dir"
else
  info "Using existing Docker image $docker_image_tag"
fi

jobs_env="${ALPINE_JOBS:-}"
if [[ -z "$jobs_env" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    jobs_env="$(nproc)"
  elif [[ "$(uname -s)" == "Darwin" ]]; then
    jobs_env="$(sysctl -n hw.ncpu)"
  else
    jobs_env="4"
  fi
fi

status "Building gem5-X (ARM opt) inside Docker container"
container_script=$(cat <<'EOS'
set -euo pipefail

if ! command -v nproc >/dev/null 2>&1; then
  nproc() { getconf _NPROCESSORS_ONLN; }
fi

if [ -f /opt/miniconda3/etc/profile.d/conda.sh ]; then
  source /opt/miniconda3/etc/profile.d/conda.sh
  conda activate py27 >/dev/null 2>&1 || conda activate py27
  export PATH=/opt/miniconda3/envs/py27/bin:$PATH
fi

real_gcc=$(command -v gcc-7 || command -v gcc)
real_gxx=$(command -v g++-7 || command -v g++)
wrapper_dir=/tmp/alpine-toolchain
mkdir -p "$wrapper_dir"
cat >"$wrapper_dir/generic-wrapper" <<'WRAP'
#!/usr/bin/env bash
real_compiler="$1"
shift
args=()
for arg in "$@"; do
  if [[ "$arg" == -Werror || "$arg" == -pedantic-errors || "$arg" == -Werror=* ]]; then
    continue
  fi
  args+=("$arg")
done
exec "$real_compiler" "${args[@]}"
WRAP
chmod +x "$wrapper_dir/generic-wrapper"
cat >"$wrapper_dir/gcc" <<WRAP2
#!/usr/bin/env bash
exec "$wrapper_dir/generic-wrapper" "$real_gcc" "\$@"
WRAP2
cat >"$wrapper_dir/g++" <<WRAP3
#!/usr/bin/env bash
exec "$wrapper_dir/generic-wrapper" "$real_gxx" "\$@"
WRAP3
chmod +x "$wrapper_dir/gcc" "$wrapper_dir/g++"
export CC="$wrapper_dir/gcc"
export CXX="$wrapper_dir/g++"

cd /project/ALPINE/gem5-X-ALPINE

if [[ "${ALPINE_CLEAN:-0}" -eq 1 ]]; then
  scons -c || true
fi

jobs="${ALPINE_JOBS:-}"
if [[ -z "$jobs" ]]; then
  jobs="$(nproc)"
fi

scons build/ARM/gem5.opt -j "$jobs"
EOS
)

docker run --rm $docker_platform_arg \
  -u "$(id -u)":"$(id -g)" \
  -e ALPINE_JOBS="$jobs_env" \
  -e ALPINE_CLEAN="${ALPINE_CLEAN:-0}" \
  -v "$alpine_src_dir":/project/ALPINE \
  -w /project/ALPINE/gem5-X-ALPINE \
  "$docker_image_tag" \
  bash -lc "$container_script"

status "gem5 binary available at: $alpine_src_dir/gem5-X-ALPINE/build/ARM/gem5.opt"
