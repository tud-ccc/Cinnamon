
#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <perfcounter.h>
#include <barrier.h>

#include "../support/common.h"

__host dpu_arguments_t DPU_INPUT_ARGUMENTS;

// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);


static void histogram(uint32_t* histo, uint32_t bins, T *input, uint32_t count){
    for(unsigned int j = 0; j < count; j++) {
        int d = ((input[j] * bins) >> DEPTH);
        histo[d] ++;
    }
}

void loadRow(int *buffer, uint32_t mem_addr, uint32_t offset, int size){
	mram_read((__mram_ptr void const*) (mem_addr + offset), buffer,  size);
}

void storeRow(int *buffer, uint32_t mem_addr, uint32_t offset, int size){
    mram_write( buffer , (__mram_ptr void *) (mem_addr + offset), size * sizeof(int));
}


int main(void) { 
    unsigned int tasklet_id = me();
    if (tasklet_id == 0){ 
        mem_reset(); 
    }

    barrier_wait(&my_barrier);

    uint32_t input_size = DPU_INPUT_ARGUMENTS.size;
    uint32_t bins = DPU_INPUT_ARGUMENTS.bins;
    uint32_t BUFFER_DIM = DPU_INPUT_ARGUMENTS.buffer_size;
    uint32_t size_per_tasklet = input_size / NR_TASKLETS;

    uint32_t base_tasklet = tasklet_id * size_per_tasklet;
    uint32_t mram_base_addr_A = (uint32_t)DPU_MRAM_HEAP_POINTER;
    uint32_t mram_base_addr_histo = (uint32_t)(DPU_MRAM_HEAP_POINTER + input_size * sizeof(T));

    uint32_t curr_mram_input_addr = mram_base_addr_A + tasklet_id * size_per_tasklet * sizeof(T);

    T *cache_input = (T *) mem_alloc(BUFFER_DIM * sizeof(T));
    T *cache_histo = (T *) mem_alloc(bins * sizeof(int));

    for(int i = 0; i < bins; i++){
        cache_histo[i] = 0;
    }

    uint32_t offset = 0 ;

    for(unsigned int i = 0 ; i < size_per_tasklet ; i += BUFFER_DIM){
        
        loadRow(cache_input, curr_mram_input_addr, offset, BUFFER_DIM * sizeof(T));
        histogram(cache_histo, bins, cache_input, BUFFER_DIM);
        offset += BUFFER_DIM * sizeof(T);
    }

    storeRow(cache_histo, mram_base_addr_histo, tasklet_id * bins * sizeof(T), bins * sizeof(T));
    return 0;
}
