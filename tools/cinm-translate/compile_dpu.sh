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

# The stack size in the header is an estimate: the kernel's buffers plus a
# fixed reserve for saved registers and spills (kStackReserveBytes in
# UPMEMOccupancy.h). A DPU has no stack guard, so a frame that outgrows it
# silently overwrites the next tasklet's stack and the program computes wrong
# answers rather than crashing. With CINM_DPU_STACK_CHECK set, the SDK's stack
# analyzer measures the linked binary's deepest frame and the compile fails
# when it exceeds what was declared. Off by default because it costs a
# disassembly per binary, which the search's compile volume would notice.
check_stack() {
    local bin_path="$1" declared="$2"
    local analyzer="$UPMEM_HOME/bin/dpu_stack_analyzer"
    local report
    report="$("$analyzer" --objdump "$UPMEM_HOME/bin/llvm-objdump" "$bin_path" 2>&1)" || {
        echo "$bin_path: dpu_stack_analyzer failed:" >&2
        echo "$report" >&2
        exit 1
    }
    local need
    need="$(echo "$report" | sed -n 's/^Max size: \([0-9]*\).*/\1/p' | head -n 1)"
    if [ -z "$need" ]; then
        echo "$bin_path: could not read 'Max size' from dpu_stack_analyzer:" >&2
        echo "$report" >&2
        exit 1
    fi
    echo "$bin_path: stack need $need of $declared bytes declared"
    if [ "$need" -gt "$declared" ]; then
        echo "$bin_path: the deepest frame needs $need bytes but the tasklet" \
             "stack is $declared; raise kStackReserveBytes or shrink the kernel" >&2
        exit 1
    fi
}

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
        if [ -n "${CINM_DPU_STACK_CHECK:-}" ]; then
            check_stack "$bin_path" "$stack_size"
        fi
        compiled=$((compiled + 1))
    fi
done

# A header that parses but names nothing is not necessarily wrong -- a module
# whose kernels were all folded away still gets translated -- so say so and
# leave it to the caller, which knows whether it expected a binary.
if [[ "$compiled" -eq 0 ]]; then
    echo "warning: $PROG: its UPMEM-TRANSLATE header names no kernel: $rest" >&2
fi
