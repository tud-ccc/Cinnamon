// Map DPU index (r,d,t) to base index in input tensor.
// :: WG -> T
#scatter_map = #affine_map<(r, d, t) -> (r, d, t * 64)>
// Reconstruct linear map
// :: Lin(WG) -> WG
#to_structured = #affine_map<(i) -> (i floordiv (16*64) mod 2, i floordiv 16 mod 64, i mod 16)>
// Layout map of the tensor
// :: T -> Lin(T)
#to_linear_t = #affine_map<(r, d, t) -> r * 8192 * 64 + d * 8192 + t>
// 

// TODO we actually need something that maps linear index 
//  to another linear index.

// :: Lin(WG) -> Lin(T)
//   to_structured . scatter_map . to_linear_t


module {
  // generate foo
  func.func @foo(%a:index) -> index {
    %res = affine.apply #scatter_map(%a): index
  }

  func.func @run_va(%A : memref<2x64x8192xi32>, %B: memref<2x64x8192xi32>, %C: memref<2x64x8192xi32>) {
    %cst00 = arith.constant 0 : index
    %cst10 = arith.constant 1 : index
    %rank_count = arith.constant 2 : index
    %dpu_count = arith.constant 64: index
    %tasklet_count = arith.constant 16 : index
    %upmem_token = upmem.alloc_dpus : !upmem.hierarchy<2x64x16>
    %base_offset = upmem.base_dpu_mem_offset : index
    %A_offset = upmem.scatter %A[64, #scatter_map] into %upmem_token at %base_offset : memref<2x64x8192xi32>, !upmem.hierarchy<2x64x16>, index -> index
    %B_offset = upmem.scatter %B[64, #scatter_map] into %upmem_token at %A_offset : memref<2x64x8192xi32>, !upmem.hierarchy<2x64x16>, index -> index
    upmem.launch %upmem_token  ranks(%arg0 upto %rank_count) dpus(%arg1 upto %dpu_count) tasklets(%arg2 upto %tasklet_count) on !upmem.hierarchy<2x64x16> {
        %cst0 = arith.constant 0 : index
        %cst1 = arith.constant 1 : index

        %ITER_I = arith.constant 128: index
        %ITER_J = arith.constant 64 : index
        %A_SIZE = arith.muli %ITER_I, %ITER_J : index
        %MRAM_ADDR = upmem.dpu_heap_base_addr : index
        %tasklet_offset = arith.muli %ITER_J, %arg2 : index
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
        upmem.terminator
    }
    %C_offset = upmem.gather %C from %upmem_token at %base_offset : memref<2x64x8192xi32>, !upmem.hierarchy<2x64x16>, index -> index
    return
  }
}


