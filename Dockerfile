FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    CC=clang \
    CXX=clang++ \
    LDFLAGS=-fuse-ld=mold \
    CMAKE_GENERATOR=Ninja

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    clang \
    cmake \
    ninja-build \
    mold \
    git \
    wget \
    curl \
    ca-certificates \
    pkg-config \
    libvulkan-dev \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    docker.io \
 && rm -rf /var/lib/apt/lists/*

# Run with -u $(id -u):$(id -g) -e HOME=/workspace so that this container and
# the inner ALPINE container (build-alpine.sh) both operate as the same host user.
ENV HOME=/workspace
WORKDIR /workspace
