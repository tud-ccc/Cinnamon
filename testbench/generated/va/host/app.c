
#include <assert.h>
#include <dpu.h>
#include <dpu_log.h>
#include <stdio.h>
#include "../support/common.h"
#include "../support/timer.h"
#ifndef DPU_BINARY
#define DPU_BINARY "./bin/dpu"
#endif

void executeVecadd(int iter, int m, int dimm, int simulation, unsigned int n_reps, unsigned int n_warmup, int buffer_size);

static T* A;
static T* B;
static T* C;

static void init_data(T* A, T* B, uint32_t m_size) {
	srand(0);

	for (unsigned int i = 0; i < m_size ; i++)
	{
		while((A[i] = (unsigned int) (rand()%50))==0);
		while((B[i] = (unsigned int) (rand()%50))==0);
	}

}


int main() {

	unsigned int reps = 3;
	unsigned int warmup = 3;

	uint32_t simulation = 0;

	// unsigned int reps = 1;
	// unsigned int warmup = 0;

	// uint32_t simulation = 1;

	uint32_t DIMSIZE;
	uint32_t dimm;
	uint32_t iter;
	uint32_t m_size;
	uint32_t n_size;
	uint32_t k_size;
	
	// DIMM 4 Non-opt
	dimm = 4;
	iter = 2;
	m_size = 2097152;
	printf("Non-opt %d\n", dimm);
	executeVecadd(iter, m_size, dimm, simulation, reps, warmup, 16);

	// DIMM 4opt
	dimm = 4;
	iter = 2;
	m_size = 2097152;
	printf("Opt %d\n", dimm);
	executeVecadd(iter, m_size, dimm, simulation, reps, warmup, 512);

	// DIMM 8 Non-opt
	dimm = 8;
	iter = 1;
	m_size = 2097152;
	printf("Non-opt %d\n", dimm);
	executeVecadd(iter, m_size, dimm, simulation, reps, warmup, 16);

	// DIMM 8 Non-opt
	dimm = 8;
	iter = 1;
	m_size = 2097152;
	printf("Opt %d\n", dimm);
	executeVecadd(iter, m_size, dimm, simulation, reps, warmup, 512);


	// DIMM 16 Non-opt
	dimm = 16;
	iter = 1;
	m_size = 1048576;
	printf("Non-opt %d\n", dimm);
	executeVecadd(iter, m_size, dimm, simulation, reps, warmup, 16);

	// DIMM 16 opt
	dimm = 16;
	iter = 1;
	m_size = 1048576;
	printf("Opt %d\n", dimm);
	executeVecadd(iter, m_size, dimm, simulation, reps, warmup, 512);

	return 0;
}


void executeVecadd(int iter, int m, int dimm, int simulation, unsigned int n_reps, unsigned int n_warmup, int buffer_size){
	uint32_t dpu_per_dimm = 128;

	uint32_t allocate_this_dpus = dpu_per_dimm * dimm;
	if(simulation == 1){
		dpu_per_dimm = 1;
		allocate_this_dpus = 1;
	}

	struct dpu_set_t dpu_set, dpu;
	unsigned int i;
	uint32_t nr_of_dpus;
	DPU_ASSERT(dpu_alloc(allocate_this_dpus, NULL, &dpu_set));
	DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));
	DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_of_dpus));

	dpu_info = (struct dpu_info_t *) malloc(nr_of_dpus * sizeof(struct dpu_info_t));
	dpu_arguments_t *input_args = (dpu_arguments_t *) malloc(nr_of_dpus * sizeof(dpu_arguments_t));


	i = 0;
	DPU_FOREACH(dpu_set, dpu, i) {
		dpu_info[i].m_size= m;
		dpu_info[i].buffer_size = buffer_size;
	}
	
	A = malloc(m * sizeof(T));
	B = malloc(m * sizeof(T));

	init_data(A, B, m);

	Timer timer;

	for (unsigned int rep = 0; rep < n_warmup + n_reps; rep++) {

		if (rep >= n_warmup)
			start(&timer, 1, rep - n_warmup);
		i = 0;
		DPU_FOREACH(dpu_set, dpu, i) {
			input_args[i].m_size = m;
			input_args[i].buffer_size = buffer_size;
			DPU_ASSERT(dpu_prepare_xfer(dpu, input_args));
		}

		DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0, sizeof(dpu_arguments_t), DPU_XFER_DEFAULT));

		i = 0;
		DPU_FOREACH(dpu_set, dpu, i) {
			DPU_ASSERT(dpu_prepare_xfer(dpu, A));
		}
		DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, m * sizeof(T), DPU_XFER_DEFAULT));

		DPU_FOREACH(dpu_set, dpu, i) {
			DPU_ASSERT(dpu_prepare_xfer(dpu, B));
		}

		DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, m * sizeof(T) , m * sizeof(T), DPU_XFER_DEFAULT));

		if (rep >= n_warmup)
			stop(&timer, 1);

		if (rep >= n_warmup)
		{
			start(&timer, 2, rep - n_warmup);
		}
		for (int x = 0 ; x < iter; x++)
			DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));

		if (rep >= n_warmup)
		{
			stop(&timer, 2);
		}

		dpu_results_t dpu_result;
		DPU_FOREACH (dpu_set, dpu, i) {
			DPU_ASSERT(dpu_prepare_xfer(dpu, &dpu_result));
		}

		if (rep >= n_warmup)
			start(&timer, 3, rep - n_warmup);
		i = 0;
		DPU_FOREACH(dpu_set, dpu, i) {
			DPU_ASSERT(dpu_prepare_xfer(dpu, C ));
		}

		DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, m * sizeof(T) , m * sizeof(T), DPU_XFER_DEFAULT));
		if(rep >= n_warmup)
			stop(&timer, 3);
	}
	free(A);
	free(B);


	print(&timer, 2, n_reps);
	
	DPU_ASSERT(dpu_free(dpu_set));

}
