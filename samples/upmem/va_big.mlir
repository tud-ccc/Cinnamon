module {
  func.func @run_va(%A : memref<2x32x8192xi32>, %B: memref<2x32x8192xi32>, %C: memref<2x32x8192xi32>) {
    %cst0 = arith.constant 0 : index
    %cst1 = arith.constant 1 : index
    %rank_count = arith.constant 2 : index
    %dpu_count = arith.constant 32: index
    %tasklet_count = arith.constant 16 : index
    %upmem_token = upmem.alloc_dpus : !upmem.hierarchy<2x32x16>
    %base_offset = upmem.base_dpu_mem_offset : index
    %A_offset = upmem.scatter %A into %upmem_token at %base_offset : memref<2x32x8192xi32>, !upmem.hierarchy<2x32x16>, index -> index
    %B_offset = upmem.scatter %B into %upmem_token at %A_offset : memref<2x32x8192xi32>, !upmem.hierarchy<2x32x16>, index -> index
    upmem.launch %upmem_token ranks(%arg0 upto %rank_count) dpus(%arg1 upto %dpu_count) tasklets(%arg2 upto %tasklet_count) on !upmem.hierarchy<2x32x16> {
        %ITER_I = arith.constant 16: index // number of tasklets
        %ITER_J = arith.constant 8 : index // number of chunks per tasklet
        %ITER_Z = arith.constant 64 : index // size of each chunk
        %MRAM_ADDR = upmem.dpu_heap_base_addr : index // starting point of MRAM = address of A
        %total_chunk_per_tasklet = arith.muli %ITER_J, %ITER_Z : index
        %A_SIZE = arith.muli %total_chunk_per_tasklet, %ITER_I : index
        %B_ADDR = arith.addi %A_SIZE, %MRAM_ADDR : index

        %tasklet_offset = arith.muli %total_chunk_per_tasklet, %arg2 : index
        %tasklet_A_MRAM_addr = arith.addi %tasklet_offset, %MRAM_ADDR : index
        %tasklet_B_MRAM_addr = arith.addi %tasklet_offset, %B_ADDR : index

        %A_buffer = upmem.pwram_alloc : memref<64xi32>
        %B_buffer = upmem.pwram_alloc : memref<64xi32>

        %t2:2 = scf.for %i0 = %cst0 to %ITER_J step %cst1 iter_args (%a_temp_addr = %tasklet_A_MRAM_addr, %b_temp_addr = %tasklet_B_MRAM_addr) -> (index,index) {
            upmem.memcpy mram_to_wram %A_buffer, %ITER_Z, %a_temp_addr : memref<64xi32>, index, index
            upmem.memcpy mram_to_wram %B_buffer, %ITER_Z, %b_temp_addr : memref<64xi32>, index, index
            scf.for %i1 = %cst0 to %ITER_Z step %cst1{
                %a = memref.load %A_buffer[%i1] : memref<64xi32>
                %b = memref.load %B_buffer[%i1] : memref<64xi32>
                %c = arith.addi %a, %b: i32
                memref.store %c, %A_buffer[%i1] : memref<64xi32>
            }
            upmem.memcpy wram_to_mram %A_buffer, %ITER_J, %a_temp_addr : memref<64xi32>, index, index
            %the_A_offset = arith.addi %a_temp_addr, %ITER_J : index 
            %the_B_offset = arith.addi %b_temp_addr, %ITER_J : index 
            scf.yield %the_A_offset, %the_B_offset : index, index
        }
        upmem.terminator
    }
    %C_offset = upmem.gather %C from %upmem_token at %base_offset : memref<2x32x8192xi32>, !upmem.hierarchy<2x32x16>, index -> index
    return
  }
}


