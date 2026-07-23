UPMEM_HOME ?= $(error UPMEM_HOME is not set)

ROOT   = ../../..
DPU_CC = $(UPMEM_HOME)/bin/dpu-upmem-dpurte-clang
CXX    = clang++

# Header-only indicators (progress bars), vendored via conan for the main
# build; reuse its resolved include path instead of duplicating the fetch.
INDICATORS_INCLUDE ?= $(shell sed -n 's/.*PACKAGE_FOLDER_RELWITHDEBINFO "\(.*\)".*/\1/p' \
                        $(ROOT)/build/indicators-relwithdebinfo-x86_64-data.cmake)/include

HOST_CXXFLAGS = -O2 -std=c++17 -I$(UPMEM_HOME)/include/dpu -I$(INDICATORS_INCLUDE) -pthread
HOST_LDFLAGS  = -ldpu -ldpuverbose -L$(UPMEM_HOME)/lib -pthread

BIN = bin

# One host binary per UPMEM transfer API under test (see the XFER_MODE
# doc-comment in scatter_bench.cpp); scatter_dpu.c is shared unchanged by all
# three since its MRAM buffer is already sized for the worst case.
HOST_BINS = $(BIN)/scatter_bench_sg $(BIN)/scatter_bench_block \
            $(BIN)/scatter_bench_broadcast

.PHONY: all dpu host run clean

all: dpu host

dpu: $(BIN)/scatter_dpu

$(BIN)/scatter_dpu: scatter_dpu.c | $(BIN)
	$(DPU_CC) $< -o $@

host: $(HOST_BINS)

$(BIN)/scatter_bench_sg: scatter_bench.cpp | $(BIN)
	$(CXX) $(HOST_CXXFLAGS) -DXFER_MODE=XFER_SG $< $(HOST_LDFLAGS) -o $@

$(BIN)/scatter_bench_block: scatter_bench.cpp | $(BIN)
	$(CXX) $(HOST_CXXFLAGS) -DXFER_MODE=XFER_BLOCK $< $(HOST_LDFLAGS) -o $@

$(BIN)/scatter_bench_broadcast: scatter_bench.cpp | $(BIN)
	$(CXX) $(HOST_CXXFLAGS) -DXFER_MODE=XFER_BROADCAST $< $(HOST_LDFLAGS) -o $@

$(BIN):
	mkdir -p $(BIN)

run: all
	./$(BIN)/scatter_bench_sg
	./$(BIN)/scatter_bench_block
	./$(BIN)/scatter_bench_broadcast

clean:
	rm -rf $(BIN) results.csv
