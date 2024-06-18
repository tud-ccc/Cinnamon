#!/bin/bash
# source /opt/upmem/upmem-2023.2.0-Linux-x86_64/upmem_env.sh

cd cinnamon/testbench
BENCH_NAME=va make clean
BENCH_NAME=mlp make clean
BENCH_NAME=1mm make clean
BENCH_NAME=2mm make clean
BENCH_NAME=3mm make clean

BENCH_NAME=va make
BENCH_NAME=mlp make
BENCH_NAME=1mm make
BENCH_NAME=2mm make
BENCH_NAME=3mm make
BENCH_NAME=mv make

