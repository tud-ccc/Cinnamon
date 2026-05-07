// RUN: true
// skip(RUN): cinm-opt %s | cinm-opt | FileCheck %s
// skip(RUN): cinm-opt %s --mlir-print-op-generic | cinm-opt | FileCheck %s


// CHECK-LABEL: matmul

#scatter_map = affine_map<(r,d,t) -> (t)>
#gather_map = affine_map<(r,d,t) -> (r,d)>

#upmem_platform = #upmem.platform<type=v1A, dimensions = 4x16>
#upmem_array = #upmem.array<ranks(4), dpus(16), tasklets(8), #upmem_platform>

func.func @matmul(%A: tensor<1024x1024xi32>, %B: tensor<1024x1024xi32>) -> tensor<1024x1024xi32> {

    %c0_i32 = arith.constant 0 : i32

    %generated = tensor.generate  {
    ^bb0(%i: index, %j: index):
        %row = tensor.extract_slice %A[%i, 0] [1, 1024] [1, 1] : tensor<1024x1024xi32> to tensor<1024xi32>
        %col = tensor.extract_slice %B[0, %j] [1024, 1] [1, 1] : tensor<1024x1024xi32> to tensor<1024xi32>
        %3 = arith.muli %row, %col : tensor<1024xi32>

        %shape = arith.constant dense<[64, 16]> : tensor<2xi32>
        %4 = tensor.reshape %3 (%shape) : (tensor<1024xi32>, tensor<2xi32>) -> tensor<64x16xi32>

        // === Lower reduction loops ===
        // Reduction has already been split into two stages: reduce 1024 elements into 64 sums of batch=16 elements
        // We pick a workgroup size that adds up to 64: 4x16
        %wg = cnm.workgroup { cnm.physical_dims = ["dpu", "tasklet"] } : !cnm.workgroup<#upmem_array>

        // We alloc the buffer for the batch (the 16 here is batch size)
        %A_buf = cnm.alloc() for %wg { cnm.physical_space = "global" } : !cnm.buffer<16xi32 on #upmem_array, level 0>
        cnm.scatter %4 into %A_buf[#scatter_map] of %wg : tensor<64x16xi32> into !cnm.buffer<16xi32 on #upmem_array, level 0>

        // We alloc a buffer for the reduction result (scalar)
        %outbuf = cnm.alloc() for %wg { cnm.physical_space = "global" } : !cnm.buffer<i32 on #upmem_array, level 0>
        // Then we launch the group
        cnm.launch %wg in(%A_buf: !cnm.buffer<16xi32 on #upmem_array, level 0>) out(%outbuf : !cnm.buffer<i32 on #upmem_array, level 0>) on !cnm.workgroup<#upmem_array> {
            ^bb0(%arg0: memref<16xi32>, %arg1: memref<i32>):
            %c0 = arith.constant 0 : i32
            // Here we have an affine reduction loop
            %total = affine.for %x = 0 to 16 iter_args(%sum = %c0) -> i32 {
                %elt = affine.load %arg0[%x]: memref<16xi32>
                %tmp = arith.addi %sum, %elt: i32
                affine.yield %tmp: i32
            }
            // finally store result
            memref.store %total, %arg1[] : memref<i32>
        }

        // Finally gather results into a buffer with same shape as the workgroup
        %r0 = tensor.empty(): tensor<4x16xi32>
        %ReductionStage1 = cnm.gather %outbuf[#gather_map] of %wg into %r0 : !cnm.buffer<i32 on #upmem_array, level 0> into tensor<4x16xi32>

        // === Second reduction loop ===
        // At this point there is a second linalg.reduce
        // I think we can always assume we do this reduction on the host.
        // Lower it to affine with --linalg-bufferize --convert-linalg-to-affine-loops

        %from_elements = tensor.from_elements %c0_i32 : tensor<i32>
        %reduced = linalg.reduce ins(%ReductionStage1 : tensor<4x16xi32>) outs(%from_elements : tensor<i32>) dimensions = [0, 1]
        (%in: i32, %init: i32) {
            %5 = arith.addi %in, %init : i32
            linalg.yield %5 : i32
        }
        %extracted = tensor.extract %reduced[] : tensor<i32>
        tensor.yield %extracted : i32
    } : tensor<1024x1024xi32>
    return %generated : tensor<1024x1024xi32>
}
