#!/bin/bash
# source /opt/upmem/upmem-2023.2.0-Linux-x86_64/upmem_env.sh

cd cinnamon/testbench
just genBench va
just genBench mlp
just genBench 1mm
just genBench 2mm
just genBench 3mm
just genBench mv
