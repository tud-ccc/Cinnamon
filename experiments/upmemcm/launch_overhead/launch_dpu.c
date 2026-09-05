// The trivial DPU program of the launch-overhead sweep: every tasklet boots
// and stops, touching nothing. What a launch of this program costs is what a
// launch costs when the kernel is free, which is the term the cost model
// leaves out -- it prices instructions and DMAs, and a program with neither
// should therefore be predicted at zero.
//
// Compiled once per tasklet count (NR_TASKLETS, see the Makefile) and reused
// for every DPU count in the sweep, so nothing about the program varies
// across the points being compared.

int main() { return 0; }
