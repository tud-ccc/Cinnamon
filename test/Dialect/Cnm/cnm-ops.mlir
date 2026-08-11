// RUN: cinm-opt %s | cinm-opt | FileCheck %s
// RUN: cinm-opt %s --mlir-print-op-generic | cinm-opt | FileCheck %s


#map = affine_map<(d0, d1) -> (d0 mod 4)>
#map1 = affine_map<(d0, d1) -> (d0 mod 128)>
#map2 = affine_map<(d0, d1) -> (d0 floordiv 128, d0 mod 128)>
#map3 = affine_map<(d0, d1) -> (d0 mod 8)>
#map4 = affine_map<(d0, d1) -> (d0 mod 16)>
#map5 = affine_map<(d0, d1) -> (d0 mod 64)>
#map6 = affine_map<(d0, d1) -> (d0 floordiv 64, d0 mod 64)>
#map7 = affine_map<(d0, d1) -> (d0 mod 2)>
#map8 = affine_map<(d0, d1) -> (d0 mod 512)>
#map9 = affine_map<(d0, d1) -> (d0 floordiv 512, d0 mod 512)>
#map10 = affine_map<(d0) -> (d0)>
#map11 = affine_map<(d0) -> ()>
#upmem = #upmem.platform<type = v1A, dpus = 4096, tasklets = 1>
#upmem_16_64_1 = #upmem.array<1024x1, #upmem>
#upmem_4_128_1 = #upmem.array<512x1, #upmem>
#upmem_8_128_1 = #upmem.array<1024x1, #upmem>

// CHECK-LABEL: @mm_dimm4_nopt
func.func @mm_dimm4_nopt(%arg0: tensor<8x1024xi32>, %arg1: tensor<1024x256xi32>) -> tensor<8x256xi32> {
    %cst = arith.constant dense<0> : tensor<4x128xi32>
    %0 = cnm.workgroup : !cnm.workgroup<#upmem_4_128_1>
    %cnm_buf = cnm.declare_buffer() for %0 : !cnm.buffer<i32 on #upmem_4_128_1>
    %cnm_buf_0 = cnm.declare_buffer() for %0 : !cnm.buffer<1024xi32 on #upmem_4_128_1>
    %cnm_buf_1 = cnm.declare_buffer() for %0 : !cnm.buffer<1024xi32 on #upmem_4_128_1>
    %1 = cinm.compute on accelerator #upmem.array<512x1, <type = v1A, dpus = 4096, tasklets = 1>> -> tensor<8x256xi32> {
      %2 = tensor.empty() : tensor<8x256xi32>
      %3 = affine.for %i = 0 to 8 step 4 iter_args(%acc = %2) -> (tensor<8x256xi32>) {
        %4 = affine.for %i_2 = 0 to 256 step 128 iter_args(%acc_3 = %acc) -> (tensor<8x256xi32>) {
          %inserted_slice = tensor.insert_slice %cst into %acc_3[%i, %i_2] [4, 128] [1, 1] : tensor<4x128xi32> into tensor<8x256xi32>
          %5 = affine.for %i_4 = 0 to 1024 step 1024 iter_args(%acc_5 = %inserted_slice) -> (tensor<8x256xi32>) {
            %extracted_slice = tensor.extract_slice %arg0[%i, %i_4] [4, 1024] [1, 1] : tensor<8x1024xi32> to tensor<4x1024xi32>
            %extracted_slice_6 = tensor.extract_slice %arg1[%i_4, %i_2] [1024, 128] [1, 1] : tensor<1024x256xi32> to tensor<1024x128xi32>
            %extracted_slice_7 = tensor.extract_slice %acc_5[%i, %i_2] [4, 128] [1, 1] : tensor<8x256xi32> to tensor<4x128xi32>
            %6 = tensor.empty() : tensor<128x1024xi32>
            %transposed = linalg.transpose ins(%extracted_slice_6 : tensor<1024x128xi32>) outs(%6 : tensor<128x1024xi32>) permutation = [1, 0]
            cnm.scatter %extracted_slice into %cnm_buf_1[#map] of %0 : tensor<4x1024xi32> into !cnm.buffer<1024xi32 on #upmem_4_128_1>
            cnm.scatter %transposed into %cnm_buf_0[#map1] of %0 : tensor<128x1024xi32> into !cnm.buffer<1024xi32 on #upmem_4_128_1>
            cnm.scatter %extracted_slice_7 into %cnm_buf[#map2] of %0 : tensor<4x128xi32> into !cnm.buffer<i32 on #upmem_4_128_1>
            cnm.launch %0 ins(%arg2 = %cnm_buf_1 : <1024xi32>, %arg3 = %cnm_buf_0 : <1024xi32>) outs(%arg4 = %cnm_buf : <i32>) on !cnm.workgroup<#upmem_4_128_1> {
              linalg.contract indexing_maps = [#map10, #map10, #map11] ins(%arg2, %arg3 : memref<1024xi32>, memref<1024xi32>) outs(%arg4 : memref<i32>)
            }
            %7 = cnm.gather %cnm_buf[#map2] of %0 into %extracted_slice_7 : !cnm.buffer<i32 on #upmem_4_128_1> into tensor<4x128xi32>
            %8 = bufferization.materialize_in_destination %7 in %extracted_slice_7 : (tensor<4x128xi32>, tensor<4x128xi32>) -> tensor<4x128xi32>
            %inserted_slice_8 = tensor.insert_slice %8 into %acc_5[%i, %i_2] [4, 128] [1, 1] : tensor<4x128xi32> into tensor<8x256xi32>
            affine.yield %inserted_slice_8 : tensor<8x256xi32>
          }
          affine.yield %5 : tensor<8x256xi32>
        }
        affine.yield %4 : tensor<8x256xi32>
      }
      cinm.yield %3 : tensor<8x256xi32>
    }
    cnm.free_workgroup %0 : !cnm.workgroup<#upmem_4_128_1>
    return %1 : tensor<8x256xi32>
  }


  memref.global "private" constant @__constant_4x128xi32 : memref<4x128xi32> = dense<0> {alignment = 64 : i64}

  func.func @mm_dimm4_opt(%arg0: memref<16x1024xi32>, %arg1: memref<1024x128xi32>, %arg2: memref<16x128xi32>) {
    %0 = memref.get_global @__constant_4x128xi32 : memref<4x128xi32>
    %1 = cnm.workgroup : !cnm.workgroup<#upmem_4_128_1>
    %cnm_buf = cnm.declare_buffer() for %1 : !cnm.buffer<i32 on #upmem_4_128_1>
    %cnm_buf_0 = cnm.declare_buffer() for %1 : !cnm.buffer<1024xi32 on #upmem_4_128_1>
    %cnm_buf_1 = cnm.declare_buffer() for %1 : !cnm.buffer<1024xi32 on #upmem_4_128_1>
    cinm.compute on accelerator #upmem.array<512x1, <type = v1A, dpus = 4096, tasklets = 1>> {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<128x1024xi32>
      affine.for %i = 0 to 16 step 4 {
        affine.for %i_2 = 0 to 128 step 128 {
          %subview = memref.subview %arg2[%i, %i_2] [4, 128] [1, 1] : memref<16x128xi32> to memref<4x128xi32, strided<[128, 1], offset: ?>>
          memref.copy %0, %subview : memref<4x128xi32> to memref<4x128xi32, strided<[128, 1], offset: ?>>
          affine.for %i_3 = 0 to 1024 step 1024 {
            %subview_4 = memref.subview %arg0[%i, %i_3] [4, 1024] [1, 1] : memref<16x1024xi32> to memref<4x1024xi32, strided<[1024, 1], offset: ?>>
            %subview_5 = memref.subview %arg1[%i_3, %i_2] [1024, 128] [1, 1] : memref<1024x128xi32> to memref<1024x128xi32, strided<[128, 1], offset: ?>>
            affine.for %i_6 = 0 to 128 {
              affine.for %i_7 = 0 to 1024 {
                %2 = affine.load %subview_5[%i_7, %i_6] : memref<1024x128xi32, strided<[128, 1], offset: ?>>
                affine.store %2, %alloc[%i_6, %i_7] : memref<128x1024xi32>
              }
            }
            cnm.scatter %subview_4 into %cnm_buf_1[#map] of %1 : memref<4x1024xi32, strided<[1024, 1], offset: ?>> into !cnm.buffer<1024xi32 on #upmem_4_128_1>
            cnm.scatter %alloc into %cnm_buf_0[#map1] of %1 : memref<128x1024xi32> into !cnm.buffer<1024xi32 on #upmem_4_128_1>
            cnm.scatter %subview into %cnm_buf[#map2] of %1 : memref<4x128xi32, strided<[128, 1], offset: ?>> into !cnm.buffer<i32 on #upmem_4_128_1>
            cnm.launch %1 ins(%arg3 = %cnm_buf_1 : <1024xi32>, %arg4 = %cnm_buf_0 : <1024xi32>) outs(%arg5 = %cnm_buf : <i32>) on !cnm.workgroup<#upmem_4_128_1> {
              affine.for %i_6 = 0 to 1024 {
                %2 = affine.load %arg3[%i_6] : memref<1024xi32>
                %3 = affine.load %arg4[%i_6] : memref<1024xi32>
                %4 = affine.load %arg5[] : memref<i32>
                %5 = arith.muli %2, %3 : i32
                %6 = arith.addi %4, %5 : i32
                affine.store %6, %arg5[] : memref<i32>
              }
            }
            cnm.gather %cnm_buf[#map2] of %1 into %subview : !cnm.buffer<i32 on #upmem_4_128_1> into memref<4x128xi32, strided<[128, 1], offset: ?>>
          }
        }
      }
      cinm.yield
    }
    cnm.free_workgroup %1 : !cnm.workgroup<#upmem_4_128_1>
    return
  }
