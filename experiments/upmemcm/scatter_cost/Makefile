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

.PHONY: all dpu host run clean

all: dpu host

dpu: $(BIN)/scatter_dpu

$(BIN)/scatter_dpu: scatter_dpu.c | $(BIN)
	$(DPU_CC) $< -o $@

host: $(BIN)/scatter_bench

$(BIN)/scatter_bench: scatter_bench.cpp | $(BIN)
	$(CXX) $(HOST_CXXFLAGS) $< $(HOST_LDFLAGS) -o $@

$(BIN):
	mkdir -p $(BIN)

run: all
	./$(BIN)/scatter_bench

clean:
	rm -rf $(BIN) results.csv
