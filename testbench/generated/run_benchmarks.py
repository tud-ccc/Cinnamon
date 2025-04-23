import os
import subprocess
import csv


def run(bench):
    os.chdir(bench)

    subprocess.run(["make", "clean"], stdout=subprocess.PIPE)
    subprocess.run(["make"])
    output = subprocess.run(["./bin/host"], stdout=subprocess.PIPE).stdout.decode('utf-8')
    os.chdir("../")
    results = {}
    last_dimm_count = None
    last_exec_type = None
    for line in output.split("\n"):
        if line.startswith("Non-opt") or line.startswith("Opt"):
            dimm_count = int(line.split(" ")[1])
            exec_type = line.split(" ")[0]
            if dimm_count not in results :
                results[dimm_count] = {}
            if exec_type not in results[dimm_count]:
                results[dimm_count][exec_type] = 0  
            last_dimm_count = dimm_count
            last_exec_type = exec_type
        
        elif len(line) > 1:
            results[last_dimm_count][last_exec_type] += float(line)
    return results





def run_cinm():
    numbers = {}          
    benchmarks = ["1mm", "2mm", "3mm", "hst", "va", "red", "sel", "mv", "conv"]            
    # benchmarks = ["red", "sel", "mv", "hst", "va"]
    for bench in benchmarks:
        print(bench)
        numbers[bench] = run(bench)
    return numbers