UPMEM_HOME ?= $(error UPMEM_HOME is not set)

DPU_CC = $(UPMEM_HOME)/bin/dpu-upmem-dpurte-clang
CXX    = clang++

HOST_CXXFLAGS = -O2 -std=c++17 -I$(UPMEM_HOME)/include/dpu
HOST_LDFLAGS  = -ldpu -ldpuverbose -L$(UPMEM_HOME)/lib

# The tasklet count the DPU program is built for. The host binary only writes
# it into the CSV, so dodo.py passes the same value to both -- change it there
# rather than here.
TASKLETS ?= 8

BIN = bin

.PHONY: all dpu host run clean

all: dpu host

dpu: $(BIN)/launch_dpu
host: $(BIN)/launch_bench

$(BIN)/launch_dpu: launch_dpu.c | $(BIN)
	$(DPU_CC) -DNR_TASKLETS=$(TASKLETS) -O3 $< -o $@

$(BIN)/launch_bench: launch_bench.cpp | $(BIN)
	$(CXX) $(HOST_CXXFLAGS) $< $(HOST_LDFLAGS) -o $@

$(BIN):
	mkdir -p $(BIN)

run: all
	LAUNCH_TASKLETS=$(TASKLETS) ./$(BIN)/launch_bench

clean:
	rm -rf $(BIN) results.csv
