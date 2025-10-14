module {
  upmem.module @upmem_funcs {
    upmem.func @va_kernel() kernel{
        %tasklet_id = upmem.tasklet_id : index
        %cst0 = arith.constant 0: index
        %cst1 = arith.constant 1: index
        %ITER_I = arith.constant 128: index
        %ITER_J = arith.constant 64 : index
        %A_SIZE = arith.muli %ITER_I, %ITER_J : index
        %MRAM_ADDR = upmem.dpu_heap_base_addr : index
        %tasklet_offset = arith.muli %ITER_J, %tasklet_id : index
        %A_MRAM_addr = arith.addi %tasklet_offset, %MRAM_ADDR : index
        %t = arith.addi %MRAM_ADDR, %A_SIZE: index 
        %B_MRAM_addr = arith.addi %t, %tasklet_offset: index
        %A_buffer = upmem.pwram_alloc : memref<64xi32>
        %B_buffer = upmem.pwram_alloc : memref<64xi32>

        %t2:2 = scf.for %i0 = %cst0 to %ITER_I step %cst1 iter_args (%a_temp_addr = %A_MRAM_addr, %b_temp_addr = %B_MRAM_addr) -> (index,index) {
            upmem.memcpy mram_to_wram %A_buffer, %ITER_J, %a_temp_addr : memref<64xi32>, index, index
            upmem.memcpy mram_to_wram %B_buffer, %ITER_J, %b_temp_addr : memref<64xi32>, index, index
            scf.for %i1 = %cst0 to %ITER_J step %cst1{
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
        upmem.return
    }
  }
}

