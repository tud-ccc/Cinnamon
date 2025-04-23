#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <perfcounter.h>
#include <handshake.h>
#include <barrier.h>

#include "../support/common.h"

__host dpu_arguments_t DPU_INPUT_ARGUMENTS;
__host dpu_results_t DPU_RESULTS[NR_TASKLETS];
int message[NR_TASKLETS];


void loadRow(int *buffer, uint32_t mem_addr, uint32_t offset, int size){
	mram_read((__mram_ptr void const*) (mem_addr + offset), buffer,  size);
}

void storeRow(int *buffer, uint32_t mem_addr, uint32_t offset, int size){
    mram_write( buffer , (__mram_ptr void *) (mem_addr + offset), size * sizeof(int));
}

BARRIER_INIT(my_barrier, NR_TASKLETS);

int main(void) { 
    unsigned int tasklet_id = me();
    if (tasklet_id == 0){ 
        mem_reset(); 
    }
    barrier_wait(&my_barrier);

    dpu_results_t *result = &DPU_RESULTS[tasklet_id];
    uint32_t input_size = DPU_INPUT_ARGUMENTS.size; 
    uint32_t BUFFER_DIM = DPU_INPUT_ARGUMENTS.buffer_size;

    uint32_t mram_base_addr_A = (uint32_t)DPU_MRAM_HEAP_POINTER;
    uint32_t size_per_tasklet = input_size/NR_TASKLETS;
    uint32_t curr_mram_addr = mram_base_addr_A + tasklet_id * sizeof(int) * BUFFER_DIM;

    int *cache_A = (int *) mem_alloc(BUFFER_DIM * sizeof(int));
	
    int output = 0;
    int offset = 0 ;
    for(unsigned int i = 0; i < size_per_tasklet ; i += BUFFER_DIM){
		loadRow(cache_A, curr_mram_addr, offset, BUFFER_DIM * sizeof(int));
        offset += BUFFER_DIM * sizeof(int);
        for (unsigned int j = 0; j < BUFFER_DIM; j++){
            output += cache_A[j];
        }
    }

    message[tasklet_id] = output;

    barrier_wait(&my_barrier);
    if(tasklet_id == 0){
        for (unsigned int each_tasklet = 1; each_tasklet < NR_TASKLETS; each_tasklet++){
            message[0] += message[each_tasklet];
        }
        result->output = message[tasklet_id];

    }

    return 0;
}
