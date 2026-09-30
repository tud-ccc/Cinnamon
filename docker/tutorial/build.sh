#!/usr/bin/env bash
# Builds the tutorial image from this checkout, tagged with its commit.
#
#   docker/tutorial/build.sh [extra docker build args...]
#
# The image needs the LLVM revision the submodule pins, which it cannot read
# itself: the build context carries no .git. It is taken from the committed
# tree, or from LLVM_REVISION when set -- a pointer moved but not yet committed,
# say. Anything else is passed to docker build. To publish, `docker login
# ghcr.io` and `docker push` the tags this prints.
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
image="${IMAGE:-ghcr.io/tud-ccc/cinnamon-tutorial}"
sha="$(git -C "$root" rev-parse --short=12 HEAD)"
llvm_revision="${LLVM_REVISION:-$(git -C "$root" ls-tree HEAD -- third-party/llvm | awk '$2 == "commit" { print $3 }')}"
[[ -n "$llvm_revision" ]] || { echo "cannot read the third-party/llvm submodule revision" >&2; exit 1; }
echo "LLVM revision: $llvm_revision"

docker build "$root" -f "$root/docker/tutorial/Dockerfile" \
  --build-arg "LLVM_REVISION=$llvm_revision" \
  -t "$image:$sha" -t "$image:latest" "$@"
echo "Built $image:$sha and $image:latest"
