#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <dpu.h>
#include <dpu_log.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>

#include "../support/common.h"
#include "../support/timer.h"

#ifndef DPU_BINARY
#define DPU_BINARY "./bin/dpu"
#endif


static int* A;
static int* C;
static int* C2;

static void read_input(int* A, unsigned int nr_elements) {
    for (unsigned int i = 0; i < nr_elements; i++) {
        A[i] = (int) (rand());
    }
 
}

void execute(int iter, int input_size, int dimm, int simulation, unsigned int n_reps, unsigned int n_warmup, int buffer_Size);


int main(int argc, char **argv) {
    unsigned int reps = 3;
	unsigned int warmup = 3;

	uint32_t simulation = 0;

    // unsigned int reps = 1;
	// unsigned int warmup = 0;

	// uint32_t simulation = 1;

    int dimm;
    int iter;
    int input_size;

    dimm = 4;
    iter = 2;
    input_size = 2097152;
    printf("Non-opt %d\n", dimm);
    execute(iter, input_size, dimm, simulation, reps, warmup, 16);
    printf("Opt %d\n", dimm);
    execute(iter, input_size, dimm, simulation, reps, warmup, 128);

    dimm = 8;
    iter = 1;
    input_size = 2097152;
    printf("Non-opt %d\n", dimm);
    execute(iter, input_size, dimm, simulation, reps, warmup, 16);
    printf("Opt %d\n", dimm);
    execute(iter, input_size, dimm, simulation, reps, warmup, 128);

    dimm = 16;
    iter = 1;
    input_size = 1048576;
    printf("Non-opt %d\n", dimm);
    execute(iter, input_size, dimm, simulation, reps, warmup, 16);
    printf("Opt %d\n", dimm);
    execute(iter, input_size, dimm, simulation, reps, warmup, 128);

}

void execute(int iter, int input_size, int dimm, int simulation, unsigned int n_reps, unsigned int n_warmup, int buffer_size){


    uint32_t dpu_per_dimm = 128;

	uint32_t allocate_this_dpus = dpu_per_dimm * dimm;
	if(simulation){
		dpu_per_dimm = 1;
	}

	if(simulation){
		allocate_this_dpus = 1;
	}

    struct dpu_set_t dpu_set, dpu;
    uint32_t nr_of_dpus;

    DPU_ASSERT(dpu_alloc(allocate_this_dpus, NULL, &dpu_set));
    DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));
    DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_of_dpus));

    unsigned int i = 0;
    uint32_t accum = 0;
    uint32_t total_count = 0;

    A = malloc(input_size * sizeof(int));
    C = malloc(input_size * sizeof(int));
    C2 = malloc(input_size * sizeof(int));
    int *bufferA = A;
    int *bufferC = C2;

    read_input(A, input_size);

    Timer timer;

	dpu_arguments_t *input_args = (dpu_arguments_t *) malloc(nr_of_dpus * sizeof(dpu_arguments_t));

    for(int rep = 0; rep < n_warmup + n_reps; rep++) {
        if(rep >= n_warmup)
            start(&timer, 1, rep - n_warmup);

        i = 0;
		DPU_FOREACH(dpu_set, dpu, i) {
			input_args[i].size = input_size;
			input_args[i].buffer_size = buffer_size;
			DPU_ASSERT(dpu_prepare_xfer(dpu, input_args + i));
		}

        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0, sizeof(dpu_arguments_t), DPU_XFER_DEFAULT));
        DPU_FOREACH(dpu_set, dpu, i) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, bufferA));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, input_size * sizeof(int), DPU_XFER_DEFAULT));
        if(rep >= n_warmup)
            stop(&timer, 1);

        if(rep >= n_warmup) {
            start(&timer, 2, rep - n_warmup);
        }

        for (int x = 0; x < iter; x++){
            DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
        }
        if(rep >= n_warmup) {
            stop(&timer, 2);
        }
    }

    print(&timer, 2, n_reps);


    free(A);
    free(C);
    free(C2);
    DPU_ASSERT(dpu_free(dpu_set));
	
}
