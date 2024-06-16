#!/bin/bash
scp -P 2293 -r ../cinnamon/testbench/generated/ reviewer@ios.inf.uni-osnabrueck.de:/home/reviewer/generated
ssh -p 2293 reviewer@ios.inf.uni-osnabrueck.de 
#cd /home/reviewer/generated
#source /opt/upmem/upmem-2023.2.0-Linux-x86_64/upmem_env.sh
#python3 run_benchmarks.py
#cat runtime.txt"
