#!/bin/bash

PROG=${1:?"Error: missing file argument"}
OUTPATH=${2:?"Error: missing out directory argument"}

dpuCompiler="${UPMEM_HOME:?"UPMEM_HOME is undefined"}"/bin/dpu-upmem-dpurte-clang

curdir="$(dirname "$0")"

mkdir -p "$OUTPATH"
header="$(head -n 1 "$PROG")"
pat="// UPMEM-TRANSLATE: (.*)"
if [[ "$header" =~ $pat ]]; then
    rest="${BASH_REMATCH[1]}"
    # todo split, capture
    for word in $(echo $rest | tr ';' ' '); do
    pat="(\w+)\:([0-9]+):(\w+).*"
    if [[ "$word" =~ $pat ]]; then
        var="${BASH_REMATCH[1]}"
        threads="${BASH_REMATCH[2]}"
        bin_name="${BASH_REMATCH[3]}"
        bin_path=$(realpath "$OUTPATH/$bin_name")

        command="'$dpuCompiler' -DNR_TASKLETS=$threads -D$var '$PROG' -o '$bin_path' '-I$curdir/dpu' -Wall -Wextra -Werror -Wno-unused-variable"
        echo $command
        eval "$command"
    fi
done
fi
