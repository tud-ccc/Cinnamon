#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
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

static T* A;
static unsigned int* histo_host;
static unsigned int* histo;
static unsigned int* histo_dpu_res;

void executeHisto(int DIMSIZE, int iter, int input_size, int bins, int dimm, int simulation, unsigned int n_reps, unsigned int n_warmup, int buffer_size);


static void read_input(T* A, char *file_name, int input_size) {

    char  dctFileName[100];
    FILE *File = NULL;

    unsigned short temp;
    sprintf(dctFileName, file_name);
    if((File = fopen(dctFileName, "rb")) != NULL) {
        for(unsigned int y = 0; y < input_size; y++) {
            fread(&temp, sizeof(unsigned short), 1, File);
            A[y] = (unsigned int)ByteSwap16(temp);
            if(A[y] >= 4096)
                A[y] = 4095;
        }
        fclose(File);
    } else {
        printf("%s does not exist\n", dctFileName);
        exit(1);
    }
}

int main(int argc, char **argv) {

    struct dpu_set_t dpu_set, dpu;
    uint32_t nr_of_dpus;
    
    unsigned int reps = 3;
	unsigned int warmup = 3;

	uint32_t simulation = 0;

    // unsigned int reps = 1;
	// unsigned int warmup = 0;

	// uint32_t simulation = 1;

	uint32_t DIMSIZE;
	uint32_t dimm;
	uint32_t iter;
	uint32_t input_size;
    uint32_t bin_count = 128;
    uint32_t dpu_per_dimm = 128;

    iter = 2;
    dimm = 4;
	input_size = 4194304;
    printf("Non-opt %d\n", dimm);
    executeHisto(DIMSIZE, iter, input_size, bin_count, dimm, simulation, reps, warmup, 16);
    printf("Opt %d\n", dimm);
    executeHisto(DIMSIZE, iter, input_size, bin_count, dimm, simulation, reps, warmup, 512);

    iter = 1;
    dimm = 8;
	input_size = 4194304;
    printf("Non-opt %d\n", dimm);
    executeHisto(DIMSIZE, iter, input_size, bin_count, dimm, simulation, reps, warmup, 16);
    printf("Opt %d\n", dimm);
    executeHisto(DIMSIZE, iter, input_size, bin_count, dimm, simulation, reps, warmup, 512);


    dimm = 16;
	input_size = 2097152;

    printf("Non-opt %d\n", dimm);
    executeHisto(DIMSIZE, iter, input_size, bin_count, dimm, simulation, reps, warmup, 16);
    printf("Opt %d\n", dimm);
    executeHisto(DIMSIZE, iter, input_size, bin_count, dimm, simulation, reps, warmup, 512);


}

void executeHisto(int DIMSIZE, int iter, int input_size, int bins, int dimm, int simulation, unsigned int n_reps, unsigned int n_warmup, int buffer_size){

	uint32_t dpu_per_dimm = 128;

	uint32_t allocate_this_dpus = dpu_per_dimm * dimm;
	if(simulation){
		dpu_per_dimm = 1;
		allocate_this_dpus = 1;
	}

	struct dpu_set_t dpu_set, dpu;
	unsigned int i;
	uint32_t nr_of_dpus;

    // Allocate DPUs and load binary
    DPU_ASSERT(dpu_alloc(allocate_this_dpus, NULL, &dpu_set));
    DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));
    DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_of_dpus));


    // Input/output allocation
    A = malloc(input_size * sizeof(T));
    T *bufferA = A;
    histo_host = malloc(bins * sizeof(unsigned int));
    histo = malloc(bins * sizeof(unsigned int));
    histo_dpu_res = malloc(bins * sizeof(unsigned int) * NR_TASKLETS);

    read_input(A, "./input/image_VanHateren.iml", input_size);

    Timer timer;

    for(int rep = 0; rep < n_warmup + n_reps; rep++) {
        memset(histo_host, 0, bins * sizeof(unsigned int));
        memset(histo, 0, bins * sizeof(unsigned int));

        if(rep >= n_warmup)
            start(&timer, 1, rep - n_warmup);

        i = 0;
	    dpu_arguments_t input_arguments[allocate_this_dpus];
	    for(i=0; i < nr_of_dpus; i++) {
	        input_arguments[i].size= input_size;
	        input_arguments[i].bins= bins;
            input_arguments[i].buffer_size = buffer_size;
	    }

        // Copy input arrays
        i = 0;
        DPU_FOREACH(dpu_set, dpu, i) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, &input_arguments[i]));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0, sizeof(input_arguments[0]), DPU_XFER_DEFAULT));
        DPU_FOREACH(dpu_set, dpu, i) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, bufferA));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, input_size * sizeof(T), DPU_XFER_DEFAULT));
        if(rep >= n_warmup)
            stop(&timer, 1);

        if(rep >= n_warmup) {
            start(&timer, 2, rep - n_warmup);
        }
        for (int x = 0; x < iter; x ++){
            DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
        }
        if(rep >= n_warmup) {
            stop(&timer, 2);
        }

        i = 0;
        if(rep >= n_warmup)
            start(&timer, 3, rep - n_warmup);
        DPU_FOREACH(dpu_set, dpu, i) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, histo_dpu_res));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, input_size * sizeof(T), bins * NR_TASKLETS * sizeof(unsigned int), DPU_XFER_DEFAULT));
		
        for(i = 1; i < NR_TASKLETS; i++){
            for(unsigned int j = 0; j < bins; j++){
                histo[j] += histo_dpu_res[j + i * NR_TASKLETS];
            }			
        }		
        if(rep >= n_warmup)
            stop(&timer, 3);

    }        
    
   
    print(&timer, 2, n_reps);
    free(A);
    free(histo_host);
    free(histo);
    free(histo_dpu_res);
    DPU_ASSERT(dpu_free(dpu_set));
	
}
