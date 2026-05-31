from __future__ import print_function

import optparse
import sys
import os

import m5
from m5.defines import buildEnv
from m5.objects import *
from m5.util import addToPath, fatal, warn

# Hardcode path to ALPINE gem5 configs so imports work regardless of cwd.
addToPath('/project/ALPINE/gem5-X-ALPINE/configs')

from common import Options
from common import Simulation
from common import CacheConfig
from common import CpuConfig
from common import MemConfig
from common.Caches import *

try:
    from ruby import Ruby
except Exception:
    Ruby = None


def get_processes(options):
    multiprocesses = []
    inputs = []
    outputs = []
    errouts = []
    pargs = []

    workloads = options.cmd.split(";")
    if options.input != "":
        inputs = options.input.split(";")
    if options.output != "":
        outputs = options.output.split(";")
    if options.errout != "":
        errouts = options.errout.split(";")
    if options.options != "":
        pargs = options.options.split(";")

    idx = 0
    for wrkld in workloads:
        process = Process(pid=100 + idx)
        process.executable = wrkld
        process.cwd = os.getcwd()

        if options.env:
            with open(options.env, 'r') as f:
                process.env = [line.rstrip() for line in f]

        if len(pargs) > idx:
            process.cmd = [wrkld] + pargs[idx].split()
        else:
            process.cmd = [wrkld]

        if len(inputs) > idx:
            process.input = inputs[idx]
        if len(outputs) > idx:
            process.output = outputs[idx]
        if len(errouts) > idx:
            process.errout = errouts[idx]

        multiprocesses.append(process)
        idx += 1

    if options.smt:
        assert options.cpu_type == 'DerivO3CPU'
        return multiprocesses, idx
    else:
        return multiprocesses, 1


parser = optparse.OptionParser()
Options.addCommonOptions(parser)
Options.addSEOptions(parser)

if Ruby is not None and "--ruby" in sys.argv:
    Ruby.define_options(parser)

(options, args) = parser.parse_args()

if args:
    print("Error: script doesn't take any positional arguments")
    sys.exit(1)

multiprocesses = []
numThreads = 1

if options.cmd:
    multiprocesses, numThreads = get_processes(options)
else:
    print("No workload specified. Exiting!\n", file=sys.stderr)
    sys.exit(1)


# ALPINE's Simulation.setCPUClass returns 4 values
(CurrCPUClass, test_mem_mode, FutureCPUClass, FutureCPUClass2) = Simulation.setCPUClass(options)
CurrCPUClass.numThreads = numThreads

if options.smt and options.num_cpus > 1:
    fatal("You cannot use SMT with multiple CPUs!")

np = options.num_cpus
mp0_path = multiprocesses[0].executable
system = System(
    cpu=[CurrCPUClass(cpu_id=i) for i in xrange(np)],
    mem_mode=test_mem_mode,
    mem_ranges=[AddrRange(options.mem_size)],
    cache_line_size=options.cacheline_size,
)

if numThreads > 1:
    system.multi_thread = True

system.voltage_domain = VoltageDomain(voltage=options.sys_voltage)
system.clk_domain = SrcClockDomain(clock=options.sys_clock, voltage_domain=system.voltage_domain)
system.cpu_voltage_domain = VoltageDomain()
system.cpu_clk_domain = SrcClockDomain(clock=options.cpu_clock, voltage_domain=system.cpu_voltage_domain)

for cpu in system.cpu:
    cpu.clk_domain = system.cpu_clk_domain

for i in xrange(np):
    if options.smt:
        system.cpu[i].workload = multiprocesses
    elif len(multiprocesses) == 1:
        system.cpu[i].workload = multiprocesses[0]
    else:
        system.cpu[i].workload = multiprocesses[i]

    if options.checker:
        system.cpu[i].addCheckerCpu()

    system.cpu[i].createThreads()

if options.ruby and Ruby is not None:
    Ruby.create_system(options, False, system)
    assert options.num_cpus == len(system.ruby._cpu_ports)

    system.ruby.clk_domain = SrcClockDomain(clock=options.ruby_clock, voltage_domain=system.voltage_domain)
    for i in xrange(np):
        ruby_port = system.ruby._cpu_ports[i]
        system.cpu[i].createInterruptController()
        system.cpu[i].icache_port = ruby_port.slave
        system.cpu[i].dcache_port = ruby_port.slave
else:
    MemClass = Simulation.setMemClass(options)
    system.membus = SystemXBar()
    system.system_port = system.membus.slave
    CacheConfig.config_cache(options, system)
    MemConfig.config_mem(options, system)

root = Root(full_system=False, system=system)
Simulation.run(options, root, system, FutureCPUClass, FutureCPUClass2)
system.workload = SEWorkload.init_compatible(mp0_path)
