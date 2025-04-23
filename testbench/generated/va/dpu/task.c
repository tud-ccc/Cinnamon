
#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <perfcounter.h>
#include <barrier.h>

#include "../support/common.h"

__host dpu_arguments_t DPU_INPUT_ARGUMENTS;

static void vector_addition(T *bufferB, T *bufferA, unsigned int l_size) {
    for (unsigned int i = 0; i < l_size; i++){
        bufferB[i] += bufferA[i];
    }
}

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

    int m_size = DPU_INPUT_ARGUMENTS.m_size; 
    int BUFFER_DIM = DPU_INPUT_ARGUMENTS.buffer_size; 


    uint32_t mram_base_addr_A = (uint32_t)DPU_MRAM_HEAP_POINTER;
    uint32_t mram_base_addr_B = (uint32_t)(DPU_MRAM_HEAP_POINTER + m_size * sizeof(int));
    uint32_t per_thread_work = m_size/(NR_TASKLETS * BUFFER_DIM);
    uint32_t curr_thread_a = mram_base_addr_A + tasklet_id * per_thread_work * sizeof(int);
    uint32_t curr_thread_b = mram_base_addr_B + tasklet_id * per_thread_work * sizeof(int);


    int *cache_A = (int *) mem_alloc(BUFFER_DIM);
    int *cache_B = (int *) mem_alloc(BUFFER_DIM);
    int offset = 0 ;

    for(unsigned int i = 0; i < (per_thread_work); i++){
        
        loadRow(cache_A, curr_thread_a, offset, BUFFER_DIM);
        loadRow(cache_B, curr_thread_b, offset, BUFFER_DIM);
        offset += BUFFER_DIM * sizeof(int);
        vector_addition(cache_B, cache_A, BUFFER_DIM);
        storeRow(cache_B, curr_thread_b, offset, BUFFER_DIM);
    }


    return 0;
}
