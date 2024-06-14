#include "host_lib.h"
#include "test.h"

void executeVecadd(int m, int dpu_count);

static int* A;
static int* B;
static int* C;

// Create input arrays
static void init_data(int* A, int* B, uint32_t m_size) {
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

	uint32_t DIMSIZE;
	uint32_t dimm;
	uint32_t iter;
	uint32_t m_size;
	uint32_t n_size;
	uint32_t k_size;

	int rank = 2;
	int dpu = 64;
	m_size = 2097152;
	A = malloc(m_size * sizeof(int));
	B = malloc(m_size * sizeof(int));
	init_data(A, B, m_size);
	executeVecadd(m_size, 1);

	free(A);
	free(B);

	return 0;
}


void executeVecadd(int m, int dpu_count){
	struct dpu_set_t *dpu_set = host_dpu_alloc(dpu_count);
	size_t offset1 = upmemrt_scatter_dpu(dpu_set, A, m * sizeof(int), m * sizeof(int), 0, &base_offset);
	size_t offset2 = upmemrt_scatter_dpu(dpu_set, B, m * sizeof(int), m * sizeof(int), offset1, &base_offset);
	launchDPUs((void *)dpu_set);
	upmemrt_gather_dpu(dpu_set, A, m * sizeof(int), m * sizeof(int), 0, &base_offset);
}
