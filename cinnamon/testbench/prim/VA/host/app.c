/**
* app.c
* VA Host Application Source File
*
*/
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
#include "../support/params.h"

// Define the DPU Binary path as DPU_BINARY here
#ifndef DPU_BINARY
#define DPU_BINARY "./bin/dpu_code"
#endif

#if ENERGY
#include <dpu_probe.h>
#endif

// Pointer declaration
static T* A;
static T* B;
static T* C;
static T* C2;

// Create input arrays
static void read_input(T* A, T* B, unsigned int nr_elements) {
    srand(0);
    printf("nr_elements\t%u\t", nr_elements);
    for (unsigned int i = 0; i < nr_elements; i++) {
        A[i] = (T) (rand());
        B[i] = (T) (rand());
    }
}

// Compute output in the host
static void vector_addition_host(T* C, T* A, T* B, unsigned int nr_elements) {
    for (unsigned int i = 0; i < nr_elements; i++) {
        C[i] = A[i] + B[i];
    }
}

void run(int alloc_dpu, int input_size);

// Main of the Host Application
int main(int argc, char **argv) {

    struct Params p = input_params(argc, argv);

    run(4, 2097152);
    run(4, 2097152);

    run(8, 2097152);

    run(16, 1048576);
}

void run(int dim, int input_size){
    int alloc_dpu = 128 * dim;
    int nr_of_dpus;
    struct dpu_set_t dpu_set, dpu;
#if ENERGY
    struct dpu_probe_t probe;
    DPU_ASSERT(dpu_probe_init("energy_probe", &probe));
#endif
    // Allocate DPUs and load binary
    DPU_ASSERT(dpu_alloc(alloc_dpu, NULL, &dpu_set));
    DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));
    DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_of_dpus));
    printf("PRIM %d\n", dim);
    unsigned int i = 0;
    
    // const unsigned int input_size = p.exp == 0 ? p.input_size * nr_of_dpus : p.input_size; // Total input size (weak or strong scaling)

    const unsigned int input_size_8bytes = 
        ((input_size * sizeof(T)) % 8) != 0 ? roundup(input_size, 8) : input_size; // Input size per DPU (max.), 8-byte aligned
    const unsigned int input_size_dpu = divceil(input_size, nr_of_dpus); // Input size per DPU (max.)
    const unsigned int input_size_dpu_8bytes = 
        ((input_size_dpu * sizeof(T)) % 8) != 0 ? roundup(input_size_dpu, 8) : input_size_dpu; // Input size per DPU (max.), 8-byte aligned

    // Input/output allocation
    A = malloc(input_size_dpu_8bytes * nr_of_dpus * sizeof(T));
    B = malloc(input_size_dpu_8bytes * nr_of_dpus * sizeof(T));
    C = malloc(input_size_dpu_8bytes * nr_of_dpus * sizeof(T));
    C2 = malloc(input_size_dpu_8bytes * nr_of_dpus * sizeof(T));
    T *bufferA = A;
    T *bufferB = B;
    T *bufferC = C2;

    // Create an input file with arbitrary data
    read_input(A, B, input_size);

    // Timer declaration
    Timer timer;
    int n_warmup = 3;
    int n_reps = 3;

    // Loop over main kernel
    for(int rep = 0; rep < n_warmup + n_reps; rep++) {

        // Compute output on CPU (performance comparison and verification purposes)
        if(rep >= n_warmup)
            start(&timer, 0, rep - n_warmup);
        vector_addition_host(C, A, B, input_size);
        if(rep >= n_warmup)
            stop(&timer, 0);

        if(rep >= n_warmup)
            start(&timer, 1, rep - n_warmup);
        // Input arguments
        unsigned int kernel = 0;
        dpu_arguments_t input_arguments[alloc_dpu];
        for(i=0; i<nr_of_dpus-1; i++) {
            input_arguments[i].size=input_size_dpu_8bytes * sizeof(T); 
            input_arguments[i].transfer_size=input_size_dpu_8bytes * sizeof(T); 
            input_arguments[i].kernel=kernel;
        }
        input_arguments[nr_of_dpus-1].size=(input_size_8bytes - input_size_dpu_8bytes * (alloc_dpu-1)) * sizeof(T); 
        input_arguments[nr_of_dpus-1].transfer_size=input_size_dpu_8bytes * sizeof(T); 
        input_arguments[nr_of_dpus-1].kernel=kernel;

        // Copy input arrays
        i = 0;
        DPU_FOREACH(dpu_set, dpu, i) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, &input_arguments[i]));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0, sizeof(input_arguments[0]), DPU_XFER_DEFAULT));

        DPU_FOREACH(dpu_set, dpu, i) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, bufferA + input_size_dpu_8bytes * i));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, input_size_dpu_8bytes * sizeof(T), DPU_XFER_DEFAULT));
 
        DPU_FOREACH(dpu_set, dpu, i) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, bufferB + input_size_dpu_8bytes * i));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, input_size_dpu_8bytes * sizeof(T), input_size_dpu_8bytes * sizeof(T), DPU_XFER_DEFAULT));
        if(rep >= n_warmup)
            stop(&timer, 1);

        // Run DPU kernel
        if(rep >= n_warmup) {
            start(&timer, 2, rep - n_warmup);
            #if ENERGY
            DPU_ASSERT(dpu_probe_start(&probe));
            #endif
        }
        DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
        if(rep >= n_warmup) {
            stop(&timer, 2);
            #if ENERGY
            DPU_ASSERT(dpu_probe_stop(&probe));
            #endif
        }

        if(rep >= n_warmup)
            start(&timer, 3, rep - n_warmup);
        i = 0;
        // PARALLEL RETRIEVE TRANSFER
        DPU_FOREACH(dpu_set, dpu, i) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, bufferC + input_size_dpu_8bytes * i));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, input_size_dpu_8bytes * sizeof(T), input_size_dpu_8bytes * sizeof(T), DPU_XFER_DEFAULT));
        if(rep >= n_warmup)
            stop(&timer, 3);

    }

    // Print timing results
    // printf("CPU ");
    // print(&timer, 0, n_reps);
    // printf("CPU-DPU ");
    // print(&timer, 1, n_reps);
    // printf("DPU Kernel ");
    print(&timer, 2, n_reps);
    // printf("DPU-CPU ");
    // print(&timer, 3, n_reps);

#if ENERGY
    double energy;
    DPU_ASSERT(dpu_probe_get(&probe, DPU_ENERGY, DPU_AVERAGE, &energy));
    printf("DPU Energy (J): %f\t", energy);
#endif	

    // Check output
    bool status = true;
    for (i = 0; i < input_size; i++) {
        if(C[i] != bufferC[i]){ 
            status = false;
#if PRINT
            printf("%d: %u -- %u\n", i, C[i], bufferC[i]);
#endif
        }
    }
    if (status) {
        printf("[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] Outputs are equal\n");
    } else {
        printf("[" ANSI_COLOR_RED "ERROR" ANSI_COLOR_RESET "] Outputs differ!\n");
    }

    // Deallocation
    free(A);
    free(B);
    free(C);
    free(C2);
    DPU_ASSERT(dpu_free(dpu_set));
	
    return status ? 0 : -1;
}
