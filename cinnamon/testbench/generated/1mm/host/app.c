#include <assert.h>
#include <dpu.h>
#include <dpu_log.h>
#include <stdio.h>
#include "../support/common.h"
#include "../support/timer.h"
#ifndef DPU_BINARY
#define DPU_BINARY "./bin/dpu"
#endif

#define OPT 1
#define NOOPT 2


void executeGEMM(int, int, int, int, int, int, int, int, unsigned int, unsigned int);

static T* A;
static T* B;
static T* C;

static void init_data(T* A, T* B, uint32_t m_size, uint32_t n_size, uint32_t q_size) {
	srand(0);

	for (unsigned int i = 0; i < m_size * q_size; i++)
	{
		while((A[i] = (unsigned int) (rand()%50))==0);
	}
	for (unsigned int i = 0; i < q_size * n_size; i++)
	{
		while((B[i] = (unsigned int) (rand()%50))==0);
	}

}

void finalEvaluation();

int main() {

	finalEvaluation();
	return 0;
}


void finalEvaluation(){
	unsigned int reps = 3;
	unsigned int warmup = 2;

	uint32_t simulation = 0;

	// unsigned int reps = 1;
	// unsigned int warmup = 0;

	// uint32_t simulation = 1;

	uint32_t exec_type = 0;
	uint32_t DIMSIZE;
	uint32_t dimm;
	uint32_t iter;
	uint32_t m_size;
	uint32_t n_size;
	uint32_t k_size;

	// DIMM 4 Non-opt
	exec_type = NOOPT;
	DIMSIZE = 1024;
	dimm = 4;
	iter = 1;
	m_size = 8;
	n_size = 256;
	k_size = 1024;
	executeGEMM(exec_type, DIMSIZE, iter, m_size, n_size, k_size, dimm, simulation, reps, warmup);

	
	// DIMM 4 opt
	exec_type = OPT;
	DIMSIZE = 1024;
	dimm = 4;
	iter = 1;
	m_size = 16;
	n_size = 128;
	k_size = 1024;
	executeGEMM(exec_type, DIMSIZE, iter, m_size, n_size, k_size, dimm, simulation, reps, warmup);

	
	// DIMM 8 Non-opt
	exec_type = NOOPT;
	DIMSIZE = 1024;
	dimm = 8;
	iter = 1;
	m_size = 8;
	n_size = 128;
	k_size = 1024;
	executeGEMM(exec_type, DIMSIZE, iter, m_size, n_size, k_size, dimm, simulation, reps, warmup);

	
	// DIMM 8 opt
	exec_type = OPT;
	DIMSIZE = 1024;
	dimm = 8;
	iter = 1;
	m_size = 16;
	n_size = 64;
	k_size = 1024;
	executeGEMM(exec_type, DIMSIZE, iter, m_size, n_size, k_size, dimm, simulation, reps, warmup);

	
	// DIMM 16 Non-opt
	exec_type = NOOPT;
	DIMSIZE = 1024;
	dimm = 16;
	iter = 1;
	m_size = 8;
	n_size = 64;
	k_size = 1024;
	executeGEMM(exec_type, DIMSIZE, iter, m_size, n_size, k_size, dimm, simulation, reps, warmup);

	
	// DIMM 16 opt
	exec_type = OPT;
	DIMSIZE = 1024;
	dimm = 16;
	iter = 1;
	m_size = 16;
	n_size = 32;
	k_size = 1024;
	executeGEMM(exec_type, DIMSIZE, iter, m_size, n_size, k_size, dimm, simulation, reps, warmup);

	
}

void executeGEMM(int execType, int DIMSIZE, int iter, int m, int n, int k, int dimm, int simulation, unsigned int n_reps, unsigned int n_warmup){
	if (execType == NOOPT){
		printf("\nNon-opt ");
	} else if (execType == OPT){
		printf("\nOpt ");
	}
	// printf("DIMMSIZE %d dimm %d iter %d m_size %d n_size %d k_size %d\n", DIMSIZE, dimm, iter, m, n, k);
	printf("%d\n", dimm);
	uint32_t dpu_per_dimm = 128;

	uint32_t allocate_this_dpus = dpu_per_dimm * dimm;
	if(simulation){
		dpu_per_dimm = 1;
	}

	if(simulation){
		n_warmup = 0;
		n_reps = 1;
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
		dpu_info[i].m_size = m;
		dpu_info[i].n_size = n;
		dpu_info[i].q_size = k;
	}
	
	
	A = malloc(m * k * sizeof(T));
	B = malloc(n * k * sizeof(T));
	C = malloc(m * n * sizeof(T));

	init_data(A, B, m, n, k);

	Timer timer;

	for (unsigned int rep = 0; rep < n_warmup + n_reps; rep++) {

		if (rep >= n_warmup)
			start(&timer, 1, rep - n_warmup);
		// Input arguments
		i = 0;
		DPU_FOREACH(dpu_set, dpu, i) {
			// Copy input arguments to DPU
			input_args[i].m_size = m;
			input_args[i].n_size = n;
			input_args[i].q_size = k;
			DPU_ASSERT(dpu_prepare_xfer(dpu, input_args + i));
		}

		DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0, sizeof(dpu_arguments_t), DPU_XFER_DEFAULT));

		i = 0;
		DPU_FOREACH(dpu_set, dpu, i) {
			DPU_ASSERT(dpu_prepare_xfer(dpu, A));
		}
		DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, m * k * sizeof(T), DPU_XFER_DEFAULT));

		DPU_FOREACH(dpu_set, dpu, i) {
			DPU_ASSERT(dpu_prepare_xfer(dpu, B));
		}

		DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, m * k * sizeof(T) , k * n * sizeof(T), DPU_XFER_DEFAULT));

		if (rep >= n_warmup)
			stop(&timer, 1);

		if (rep >= n_warmup)
		{
			start(&timer, 2, rep - n_warmup);
		}

		DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));

		if (rep >= n_warmup)
		{
			stop(&timer, 2);
		}

		// // Display DPU Logs
		// DPU_FOREACH(dpu_set, dpu) {
		// 	DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
		// }

		// Retrieve results
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

		DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, m * n * sizeof(T) + n * k * sizeof(T) , m * n * sizeof(T), DPU_XFER_DEFAULT));
		if(rep >= n_warmup)
			stop(&timer, 3);
	}
	free(A);
	free(B);
	free(C);


	// Print timing results
	// printf("M %d N %d Q %d\n", m, n, k);
	// printf("CPU-DPU Time (ms): ");
	// printf("Iter %d\n",iter);
	// print(&timer, 1, n_reps);
	// printf("DPU Kernel Time (ms): ");
	print(&timer, 2, n_reps);
	// printf("DPU-CPU Time (ms): ");
	// print(&timer, 3, n_reps);
	printf("\n");
	
	DPU_ASSERT(dpu_free(dpu_set));

}