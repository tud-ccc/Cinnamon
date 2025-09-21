#!/usr/bin/env bash
set -euo pipefail

# Compile the Alpine runtime example inside the alpine-gem5 Docker image.
# Outputs an AArch64 SE binary at third-party/ALPINE/working_test/aimc_probe.out.

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
runtime_dir="$repo_root/runtime/Alpine"
output_dir="$repo_root/third-party/ALPINE/working_test"
output_bin="$output_dir/aimc_probe.out"
docker_image="${ALPINE_DOCKER_TAG:-alpine-gem5:latest}"

required=(
  "$runtime_dir/example.cc"
  "$runtime_dir/alpine_runtime.cc"
  "$runtime_dir/memref_rt_min.cc"
  "$runtime_dir/runtime_shims.cc"
  "$runtime_dir/crt0.o"
)

missing=()
for path in "${required[@]}"; do
  [[ -f "$path" ]] || missing+=("$path")
done

if ((${#missing[@]})); then
  printf 'Missing required runtime sources:\n' >&2
  for path in "${missing[@]}"; do
    printf '  %s\n' "$path" >&2
  done
  exit 1
fi

mkdir -p "$output_dir"

cat <<INFO
[host] repo root: $repo_root
[host] runtime dir: $runtime_dir
[host] output: $output_bin
[host] docker image: $docker_image
INFO

docker run --rm \
  -u "$(id -u):$(id -g)" \
  -v "$repo_root":/workspace \
  -w /workspace/runtime/Alpine \
  "$docker_image" /bin/sh -eu -c '
    CXXFLAGS="-O3 -std=c++17 -I. -DUSE_CHECKER -ffreestanding -fno-exceptions -fno-rtti -fno-pic"
    aarch64-linux-gnu-g++ $CXXFLAGS -c example.cc -o example.o
    aarch64-linux-gnu-g++ $CXXFLAGS -c alpine_runtime.cc -o alpine_runtime.o
    aarch64-linux-gnu-g++ $CXXFLAGS -c memref_rt_min.cc -o memref_rt_min.o
    aarch64-linux-gnu-g++ $CXXFLAGS -c runtime_shims.cc -o runtime_shims.o
    aarch64-linux-gnu-ld -static -nostdlib -no-pie --no-dynamic-linker \
      -e _start --defsym __dso_handle=0 \
      -o /workspace/third-party/ALPINE/working_test/aimc_probe.out \
      example.o crt0.o alpine_runtime.o runtime_shims.o memref_rt_min.o
  '

echo "Built: $output_bin"
