#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <barrier.h>
#include <seqread.h>
#include <stdlib.h>

#include "../support/common.h"
#include <perfcounter.h>

#define BUFFER_DIM 64

__host dpu_arguments_t DPU_INPUT_ARGUMENTS;

__host dpu_results_t DPU_RESULTS;

// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);


void loadSubmat(int * buffer, uint32_t mram_address, uint32_t offset, int dim){
	mram_read((__mram_ptr void const*) (mram_address + offset), buffer,  dim * sizeof(int));
}

void loadRow(int *buffer, uint32_t mem_addr, uint32_t offset, int size){
	mram_read((__mram_ptr void const*) (mem_addr + offset), buffer,  size);
}

void storeRow(int *buffer, uint32_t mem_addr, uint32_t offset, int size){
    mram_write( buffer , (__mram_ptr void *) (mem_addr + offset), size * sizeof(int));
}

int main() {

	unsigned int tasklet_id = me();
	if (tasklet_id == 0){ 
		mem_reset(); // Reset the heap
	}

	barrier_wait(&my_barrier);


    int m_size = DPU_INPUT_ARGUMENTS.m_size; // output rows
    int n_size = DPU_INPUT_ARGUMENTS.n_size; // output cols
    int q_size = DPU_INPUT_ARGUMENTS.q_size; // computation count 

    uint32_t mram_base_addr_A = (uint32_t) (DPU_MRAM_HEAP_POINTER ) ;
    uint32_t mram_base_addr_B = (uint32_t) (DPU_MRAM_HEAP_POINTER + m_size * q_size * sizeof(int)) ;
    uint32_t mram_base_addr_C = (uint32_t) (DPU_MRAM_HEAP_POINTER + (m_size * q_size + n_size * q_size) * sizeof(int));

    int32_t rows_per_thread;
    if(m_size < NR_TASKLETS){
        rows_per_thread = 1;
    } else {
        rows_per_thread = m_size / NR_TASKLETS;
    }
    int32_t columns_per_thread = n_size;

    if(tasklet_id > m_size){
		return 0;
	}

	uint32_t mram_curr_thread_A = mram_base_addr_A + tasklet_id * rows_per_thread * q_size * sizeof(T);
	uint32_t mram_curr_thread_B = mram_base_addr_B;
	uint32_t mram_curr_thread_C = mram_base_addr_C;

	int *cache_A = (int *) mem_alloc(sizeof(int) * BUFFER_DIM);
    int *cache_B = (int *) mem_alloc(sizeof(int) * BUFFER_DIM);
	int *cache_C = (int *) mem_alloc(sizeof(int) * BUFFER_DIM);

    uint32_t calc_type = (q_size <= BUFFER_DIM)? 1 : 0;
    uint32_t write_c_count = 0 ; 
	uint32_t a_loads = 0;
	uint32_t b_loads = 0;
	uint32_t mult = 0;

	uint32_t mram_load_size;
	if(calc_type == 0){
		mram_load_size = BUFFER_DIM * sizeof(int);
	} else {
		mram_load_size = q_size * sizeof(int);
	}
	uint32_t a_offset = 0;
	uint32_t b_offset = 0;        
	uint32_t c_offset = 0;        
	uint32_t load_per_row = q_size / BUFFER_DIM;
	uint32_t mram_temp_c = mram_curr_thread_C;
    uint32_t write_index = 0;

    if(calc_type == 1){
            // if(me() == 0){
            //     printf("size per thread%d\n", rows_per_thread);
            // }
        for (int i = 0 ; i < rows_per_thread; i++){
            a_offset += q_size * sizeof(int);
            loadRow(cache_A, mram_curr_thread_A, a_offset, mram_load_size);
            b_offset = 0;
            for(int j = 0 ; j < n_size; j++){
				b_offset += q_size * sizeof(int);
                loadRow(cache_B, mram_curr_thread_B, b_offset, mram_load_size); // assuming the B is already transposed 
                cache_C[j] = 0;
                for(int z = 0 ; z < q_size ; z++){
                    cache_C[j] += cache_A[z] * cache_B[z];
                }
                write_index++;
                if(write_index == BUFFER_DIM){
                    storeRow(cache_C, mram_base_addr_C, b_offset, mram_load_size);
                    c_offset += BUFFER_DIM * sizeof(int);
                }
            }
        }
    } else {
        int dim_point = BUFFER_DIM * sizeof(T);
        int write_index = 0;
            // if(me() == 0){
            //     printf("size per thread%d\n", rows_per_thread);
            // }
        for (int i = 0 ; i < rows_per_thread; i++){
			// a_offset += (q_size * sizeof(int));
            b_offset = 0;
            for(int j = 0 ; j < n_size; j++){
                for(int l = 0 ; l < load_per_row; l++){
                    loadRow(cache_A, mram_curr_thread_A, a_offset , mram_load_size);
                    loadRow(cache_B, mram_curr_thread_B, b_offset, mram_load_size); // assuming the B is already transposed 
					a_offset += BUFFER_DIM * sizeof(int);
					b_offset += BUFFER_DIM * sizeof(int);
                    cache_C[j] = 0;
                    for(int z = 0 ; z < BUFFER_DIM; z++){
                        cache_C[j] += cache_A[z] * cache_B[z];
                    }
                    write_index++;
                    if(write_index == BUFFER_DIM){
                        storeRow(cache_C, mram_base_addr_C, c_offset, mram_load_size);
                        c_offset += BUFFER_DIM * sizeof(int);
                    }
                }
    
                
            }
        }

    }

 


	return 0;
}