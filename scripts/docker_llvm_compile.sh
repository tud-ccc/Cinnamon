#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage: ${0##*/} <driver.c> <kernel.ll> [output-binary]

  driver.c        Path to the C driver source (compiled inside the container).
  kernel.ll       Path to the LLVM IR (.ll) file (compiled to .o on the host).
  output-binary   Optional output path (defaults to third-party/ALPINE/working_test/<kernel>.out).

All paths must reside within the repository so they are visible inside the Docker container.

Example:
  ${0##*/} tutorial/assets/drivers/conv2d_driver.c tutorial/artifacts/conv_add_relu_cinm10.ll ./out
USAGE
}

# Portable helpers for absolute/relative paths without GNU realpath requirements
abs_path() {
  python3 - "$1" <<'PY'
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
}

rel_to_repo() {
  python3 - "$repo_root" "$1" <<'PY'
import os, sys
root = os.path.abspath(sys.argv[1])
path = os.path.abspath(sys.argv[2])
if os.path.commonpath([root, path]) != root:
    sys.exit(1)
print(os.path.relpath(path, root))
PY
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage >&2
  exit 1
fi

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

# Basic tool checks
command -v python3 >/dev/null 2>&1 || { echo "python3 is required" >&2; exit 1; }
command -v docker  >/dev/null 2>&1 || { echo "docker is required"  >&2; exit 1; }
command -v clang   >/dev/null 2>&1 || { echo "clang (host) is required" >&2; exit 1; }

# Inputs
abs_driver=$(abs_path "$1")
abs_llvm=$(abs_path "$2")

[[ -f "$abs_driver" ]] || { echo "Driver source not found: $abs_driver" >&2; exit 1; }
[[ -f "$abs_llvm"   ]] || { echo "LLVM IR (.ll) not found: $abs_llvm" >&2; exit 1; }

case "$abs_llvm" in
  *.ll) ;;
  *)
    echo "Unsupported input extension (expected .ll): $abs_llvm" >&2
    exit 1
    ;;
esac

# Names/paths
kernel_base="$(basename "${abs_llvm%.ll}")"
build_root="$repo_root/runtime/Alpine/llvm_build"
build_dir="$build_root/$kernel_base"
llvm_obj="$build_dir/${kernel_base}.o"

if [[ $# -eq 3 ]]; then
  abs_output=$(abs_path "$3")
else
  abs_output="$repo_root/third-party/ALPINE/working_test/${kernel_base}.out"
fi

# Ensure paths live inside repo (so Docker can see them via -v mount)
mkdir -p "$build_dir" "$(dirname "$abs_output")"

if ! output_rel=$(rel_to_repo "$abs_output"); then
  echo "Output path must reside within the repository: $abs_output" >&2
  exit 1
fi
if ! driver_rel=$(rel_to_repo "$abs_driver"); then
  echo "Driver path must reside within the repository: $abs_driver" >&2
  exit 1
fi
if ! build_rel=$(rel_to_repo "$build_dir"); then
  echo "Build directory must reside within the repository: $build_dir" >&2
  exit 1
fi

# Compile LLVM IR to object on host
clang -target aarch64-linux-gnu -O3 -ffreestanding -fno-exceptions -fno-rtti -fno-pic \
  -c "$abs_llvm" -o "$llvm_obj"

# Relative path for Docker envs
llvm_obj_rel=$(rel_to_repo "$llvm_obj")

# Info
cat <<INFO
[host] repo root: $repo_root
[host] driver src: $abs_driver
[host] llvm ir: $abs_llvm
[host] llvm obj: $llvm_obj
[host] output bin: $abs_output
INFO

docker_image="${ALPINE_DOCKER_TAG:-alpine-gem5:latest}"
echo "[host] docker image: $docker_image"

# Build (compile runtime + driver, then link) inside container
cat <<'INFO'
[host] compiling and linking inside container…
INFO

# When running inside a container with a forwarded Docker socket, volume paths
# are resolved on the host. Use CINNAMON_HOST_PATH if set.
host_repo_root="${CINNAMON_HOST_PATH:-$repo_root}"

docker run --rm \
  -u "$(id -u):$(id -g)" \
  -v "$host_repo_root":/workspace \
  -w /workspace \
  -e BUILD_DIR="$build_rel" \
  -e DRIVER_SRC="$driver_rel" \
  -e LLVM_OBJ="$llvm_obj_rel" \
  -e OUTPUT_BIN="$output_rel" \
  "$docker_image" /bin/sh -eu -c '
    BUILD_DIR="/workspace/${BUILD_DIR}"
    DRIVER_SRC="/workspace/${DRIVER_SRC}"
    LLVM_OBJ="/workspace/${LLVM_OBJ}"
    OUTPUT_BIN="/workspace/${OUTPUT_BIN}"
    OUTPUT_DIR="$(dirname "$OUTPUT_BIN")"
    mkdir -p "$OUTPUT_DIR"

    CXXFLAGS="-O3 -std=c++17 -I/workspace/runtime/Alpine -DUSE_CHECKER -ffreestanding -fno-exceptions -fno-rtti -fno-pic"

    aarch64-linux-gnu-g++ $CXXFLAGS -c /workspace/runtime/Alpine/alpine_runtime.cc   -o "$BUILD_DIR/alpine_runtime.o"
    aarch64-linux-gnu-g++ $CXXFLAGS -c /workspace/runtime/Alpine/aimc_state.cc       -o "$BUILD_DIR/aimc_state.o"
    aarch64-linux-gnu-g++ $CXXFLAGS -c /workspace/runtime/Alpine/memref_rt_min.cc    -o "$BUILD_DIR/memref_rt_min.o"
    aarch64-linux-gnu-g++ $CXXFLAGS -c /workspace/runtime/Alpine/runtime_shims.cc    -o "$BUILD_DIR/runtime_shims.o"
    aarch64-linux-gnu-gcc  -O3 -std=gnu11 -ffreestanding -fno-pic -c "$DRIVER_SRC"   -o "$BUILD_DIR/driver.o"

    aarch64-linux-gnu-ld -static -nostdlib -no-pie --no-dynamic-linker \
      -e _start --defsym __dso_handle=0 \
      -o "$OUTPUT_BIN" \
      "$BUILD_DIR/driver.o" \
      "$LLVM_OBJ" \
      "$BUILD_DIR/alpine_runtime.o" \
      "$BUILD_DIR/aimc_state.o" \
      "$BUILD_DIR/runtime_shims.o" \
      "$BUILD_DIR/memref_rt_min.o" \
      /workspace/runtime/Alpine/crt0.o
  '

echo "Built: $abs_output"
