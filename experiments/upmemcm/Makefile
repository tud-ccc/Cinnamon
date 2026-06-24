UPMEM_HOME ?= $(error UPMEM_HOME is not set)

DPU_CC  = $(UPMEM_HOME)/bin/dpu-upmem-dpurte-clang
CXX     = clang++

DPU_IDIR     = ../../testbench/lib/dpu
DPU_CFLAGS   = -DSTACK_SIZE_DEFAULT=512 -I$(DPU_IDIR) -Wall -Wno-unused-variable

HOST_CXXFLAGS = -O2 -I$(UPMEM_HOME)/include/dpu
HOST_LDFLAGS  = -ldpu -ldpuverbose -L$(UPMEM_HOME)/lib

BIN = bin

TASKLET_COUNTS   = $(shell seq 1 24)
DPU_BINS         = $(patsubst %,$(BIN)/barrier_%,$(TASKLET_COUNTS))
DPU_CONTROL_BINS = $(patsubst %,$(BIN)/barrier_%_control,$(TASKLET_COUNTS))

.PHONY: all dpu host clean

all: dpu host

dpu: $(DPU_BINS) $(DPU_CONTROL_BINS)

$(BIN):
	mkdir -p $(BIN)

# Pattern rule: barrier_N compiled with -DNR_TASKLETS=N
$(BIN)/barrier_%: barrier.c | $(BIN)
	$(DPU_CC) -DNR_TASKLETS=$* -DBENCH $(DPU_CFLAGS) $< -o $@

$(BIN)/barrier_%_control: barrier.c | $(BIN)
	$(DPU_CC) -DNR_TASKLETS=$* $(DPU_CFLAGS) $< -o $@

host: $(BIN)/barrierhost

$(BIN)/barrierhost: barrierhost.cpp | $(BIN)
	$(CXX) $(HOST_CXXFLAGS) $< $(HOST_LDFLAGS) -o $@

clean:
	rm -rf $(BIN)
