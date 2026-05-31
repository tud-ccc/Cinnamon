#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
mlir_dir="$repo_root/runtime/Alpine/mlir_test"
mlir_src="$mlir_dir/alpine_example.mlir"
llvm_mlir="$mlir_dir/build/alpine_example.llvm.mlir"
llvm_ir="$mlir_dir/build/alpine_example.ll"
llvm_obj="$mlir_dir/build/alpine_example.o"
driver_src="$mlir_dir/driver.c"
output_dir="$repo_root/third-party/ALPINE/working_test"
output_bin="$output_dir/aimc_probe_mlir.out"
docker_image="${ALPINE_DOCKER_TAG:-alpine-gem5:latest}"
cinm_opt="$repo_root/build/bin/cinm-opt"
cinm_translate="$repo_root/build/bin/cinm-translate"

if [[ ! -x "$cinm_opt" ]]; then
  echo "cinm-opt not found at $cinm_opt; build the project first." >&2
  exit 1
fi
if [[ ! -x "$cinm_translate" ]]; then
  echo "cinm-translate not found at $cinm_translate; build the project first." >&2
  exit 1
fi
if [[ ! -f "$mlir_src" ]]; then
  echo "MLIR source missing at $mlir_src" >&2
  exit 1
fi
if [[ ! -f "$driver_src" ]]; then
  echo "Driver source missing at $driver_src" >&2
  exit 1
fi
if ! command -v clang >/dev/null 2>&1; then
  echo "clang is required on the host to lower LLVM IR to an object file." >&2
  exit 1
fi

mkdir -p "$mlir_dir/build" "$output_dir"

pipeline='builtin.module(convert-alpine-to-func,convert-scf-to-cf,convert-arith-to-llvm,convert-to-llvm,reconcile-unrealized-casts)'
"$cinm_opt" --pass-pipeline="$pipeline" "$mlir_src" > "$llvm_mlir"
"$cinm_translate" --mlir-to-llvmir "$llvm_mlir" > "$llvm_ir"

clang -target aarch64-linux-gnu -O3 -ffreestanding -fno-exceptions -fno-rtti \
  -fno-pic -c "$llvm_ir" -o "$llvm_obj"

cat <<INFO
[host] repo root: $repo_root
[host] mlir src: $mlir_src
[host] lowered mlir: $llvm_mlir
[host] llvm ir: $llvm_ir
[host] llvm obj: $llvm_obj
[host] output: $output_bin
[host] docker image: $docker_image
INFO

# Compile/link inside the alpine-gem5 docker image.
docker run --rm \
  -u "$(id -u):$(id -g)" \
  -v "$repo_root":/workspace \
  -w /workspace \
  "$docker_image" /bin/sh -eu -c '
    BUILD_DIR=/workspace/runtime/Alpine/mlir_test/build
    CXXFLAGS="-O3 -std=c++17 -I/workspace/runtime/Alpine -DUSE_CHECKER -ffreestanding -fno-exceptions -fno-rtti -fno-pic"
    aarch64-linux-gnu-g++ $CXXFLAGS -c /workspace/runtime/Alpine/alpine_runtime.cc -o "$BUILD_DIR/alpine_runtime.o"
    aarch64-linux-gnu-g++ $CXXFLAGS -c /workspace/runtime/Alpine/memref_rt_min.cc -o "$BUILD_DIR/memref_rt_min.o"
    aarch64-linux-gnu-g++ $CXXFLAGS -c /workspace/runtime/Alpine/runtime_shims.cc -o "$BUILD_DIR/runtime_shims.o"
    aarch64-linux-gnu-gcc -O3 -std=gnu11 -ffreestanding -fno-pic -c /workspace/runtime/Alpine/mlir_test/driver.c -o "$BUILD_DIR/driver.o"
    aarch64-linux-gnu-ld -static -nostdlib -no-pie --no-dynamic-linker \
      -e _start --defsym __dso_handle=0 \
      -o /workspace/third-party/ALPINE/working_test/aimc_probe_mlir.out \
      "$BUILD_DIR/driver.o" "$BUILD_DIR/alpine_example.o" \
      "$BUILD_DIR/alpine_runtime.o" "$BUILD_DIR/runtime_shims.o" \
      "$BUILD_DIR/memref_rt_min.o" \
      /workspace/runtime/Alpine/crt0.o
  '

echo "Built: $output_bin"
