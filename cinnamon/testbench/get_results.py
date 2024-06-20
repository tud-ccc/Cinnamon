import os
import subprocess 
import generated.run_benchmarks
import prim.run_benchmarks

results = {}
os.chdir("generated")
results["cinm"] = generated.run_benchmarks.run_cinm()
os.chdir("../prim")
results["prim"] = prim.run_benchmarks.run_prim()
os.chdir("..")

f = open("exp-fig-12.txt", "w")

dimm_counts = [4, 8, 16]
print(results)

benchmarks = {"va":"VA", "sel":"SEL", "mv":"GEMV", "hst":"HST-L", "red":"RED"}

for bench in benchmarks:
    f.write(bench + " ")
    for dimm_count in dimm_counts:
        f.write(str(results["prim"][benchmarks[bench]][dimm_count]))
        f.write(" ")
        f.write(str(results["cinm"][bench][dimm_count]["Opt"]))
        f.write(" ")
    f.write("\n")
     
f.close()