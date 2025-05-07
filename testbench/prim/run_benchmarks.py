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
        if line.startswith("PRIM"):
            dimm_count = int(line.split(" ")[1])
            if dimm_count not in results :
                results[dimm_count] = 0
            last_dimm_count = dimm_count
        
        elif len(line) > 1:
            results[last_dimm_count] += float(line)
    return results


def run_prim():

    numbers = {}          
    benchmarks = ["RED", "HST-L", "VA", "SEL", "GEMV"]
    for bench in benchmarks:
        # print(bench)
        numbers[bench] = run(bench)
    return numbers 

# for bench in benchmarks:
#     f.write(bench + " ")
#     for dimm_count in dimm_counts:
#         f.write(str(numbers[bench][dimm_count]["Non-opt"]))
#         f.write(" ")
#         f.write(str(numbers[bench][dimm_count]["Opt"]))
#         f.write(" ")
#     f.write("\n")
     
# f.close()

