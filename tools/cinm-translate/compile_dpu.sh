#!/bin/bash
#
# Compile the DPU-side C that cinm-translate emits, into one DPU binary per
# kernel the file holds.
#
# This is the reader half of the UPMEM C target's output contract. The
# translator writes, as the first line of every .dpu.c it produces,
#
#     // UPMEM-TRANSLATE: <var>:<tasklets>:<stack bytes>:<binary name>; ...
#
# one entry per kernel (see UPMEMTranslateToCpp.cpp, printCompilationVar).
# Several kernels can share one C file, each guarded by `#ifdef <var>`, so the
# file is compiled once per entry with that entry's var defined and its tasklet
# count and stack size fixed -- which is why the parameters travel in the source
# rather than on this script's command line. Nothing else parses that header, so
# this script and that emitter change together.
#
# Usage: cinm-compile-dpu <file.dpu.c> <output directory>

set -euo pipefail

PROG=${1:?"Error: missing file argument"}
OUTPATH=${2:?"Error: missing out directory argument"}

dpuCompiler="${UPMEM_HOME:?"UPMEM_HOME is undefined"}"/bin/dpu-upmem-dpurte-clang

mkdir -p "$OUTPATH"
header="$(head -n 1 "$PROG")"
pat="// UPMEM-TRANSLATE: (.*)"
if [[ ! "$header" =~ $pat ]]; then
    echo "$PROG: no '// UPMEM-TRANSLATE:' header on the first line." >&2
    echo "Was it produced by cinm-translate --mlir-to-upmem-cpp?" >&2
    exit 1
fi

rest="${BASH_REMATCH[1]}"
compiled=0
for word in $(echo "$rest" | tr ';' ' '); do
    pat="(\w+)\:([0-9]+):([0-9]+):(\w+).*"
    if [[ "$word" =~ $pat ]]; then
        var="${BASH_REMATCH[1]}"
        threads="${BASH_REMATCH[2]}"
        stack_size="${BASH_REMATCH[3]}"
        bin_name="${BASH_REMATCH[4]}"
        bin_path=$(realpath "$OUTPATH/$bin_name")

        command="'$dpuCompiler' -DSTACK_SIZE_DEFAULT=$stack_size -DNR_TASKLETS=$threads -D$var '$PROG' -o '$bin_path' -O3 -Wall -Wextra -Werror -Wno-unused-variable"
        echo "$command"
        eval "$command"
        compiled=$((compiled + 1))
    fi
done

# A header that parses but names nothing is not necessarily wrong -- a module
# whose kernels were all folded away still gets translated -- so say so and
# leave it to the caller, which knows whether it expected a binary.
if [[ "$compiled" -eq 0 ]]; then
    echo "warning: $PROG: its UPMEM-TRANSLATE header names no kernel: $rest" >&2
fi
