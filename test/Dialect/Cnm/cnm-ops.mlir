// RUN: cinm-opt %s | cinm-opt | FileCheck %s
// RUN: cinm-opt %s --mlir-print-op-generic | cinm-opt | FileCheck %s


// CHECK-LABEL: matmul

#scatter_map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#gather_map = affine_map<(d0, d1) -> (d0, d1)>

func.func @matmul(%A: tensor<1024x1024xi32>, %B: tensor<1024x1024xi32>) -> tensor<1024x1024xi32> {

    %c0_i32 = arith.constant 0 : i32

    %generated = tensor.generate  {
    ^bb0(%i: index, %j: index):
        %row = tensor.extract_slice %A[%i, 0] [1, 1024] [1, 1] : tensor<1024x1024xi32> to tensor<1024xi32>
        %col = tensor.extract_slice %B[0, %j] [1024, 1] [1, 1] : tensor<1024x1024xi32> to tensor<1024xi32>
        %3 = arith.muli %row, %col : tensor<1024xi32>

        // === Lower reduction loops ===
        // Reduction has already been split into two stages: reduce 1024 elements into 64 sums of batch=16 elements
        // We pick a workgroup size that adds up to 64: 4x16
        %wg = cnm.workgroup [4 16] { cnm.physical_dims = ["dpu", "tasklet"] }
        // We reshape the tensor to start with these dimensions
        %shape = arith.constant dense<[ 4, 16, 16 ]> : tensor<3xi32>
        %reshaped = tensor.reshape %3(%shape): (tensor<1024xi32>, tensor<3xi32>) -> tensor<4x16x16xi32>

        // We alloc the buffer for the batch (the 16 here is batch size)
        %A_buf = cnm.alloc() for %wg { cnm.physical_space = "global" } : !cnm.buffer<16xi32 on 4x16, level 0> for !cnm.workgroup<4x16>
        // Scatter. Scatter map is (d0, d1) -> (d0 floordiv WG[0], d1 floordiv WG[1], d0 mod WG[0], d1 mod WG[1])
        // where WG[0], WG[1] are dimensions of workgroup
        %sc_a_token = cnm.scatter %reshaped into %A_buf[#scatter_map] of %wg : tensor<4x16x16xi32> into !cnm.buffer<16xi32 on 4x16, level 0> of !cnm.workgroup<4x16>

        // We alloc a buffer for the reduction result (scalar)
        %outbuf = cnm.alloc() for %wg { cnm.physical_space = "global" } : !cnm.buffer<i32 on 4x16, level 0> for !cnm.workgroup<4x16>
        // Then we launch the group
        %token2 = cnm.launch %wg (%A_buf, %outbuf: !cnm.buffer<16xi32 on 4x16, level 0>, !cnm.buffer<i32 on 4x16, level 0>) on !cnm.workgroup<4x16> {
            ^bb0(%arg0: memref<16xi32>, %arg1: memref<i32>):
            // Here we have an affine reduction loop
            %total = affine.for %x = 0 to 16 iter_args(%sum = %c0_i32) -> i32 {
                %elt = affine.load %arg0[%x]: memref<16xi32>
                %tmp = arith.addi %sum, %elt: i32
                affine.yield %tmp: i32
            }
            // finally store result
            memref.store %total, %arg1[] : memref<i32>
        }

        // Finally gather results into a buffer with same shape as the workgroup
        %ReductionStage1, %token3 = cnm.gather %outbuf[#gather_map] of %wg : !cnm.buffer<i32 on 4x16, level 0> of !cnm.workgroup<4x16> into tensor<4x16xi32>

        // === Second reduction loop ===
        // At this point there is a second linalg.reduce
        // I think we can always assume we do this reduction on the host.
        // Lower it to affine with --linalg-bufferize --convert-linalg-to-affine-loops

        %from_elements = tensor.from_elements %c0_i32 : tensor<i32>
        %reduced = linalg.reduce ins(%ReductionStage1 : tensor<4x16xi32>) outs(%from_elements : tensor<i32>) dimensions = [0, 1]
        (%in: i32, %init: i32) {
            %4 = arith.addi %in, %init : i32
            linalg.yield %4 : i32
        }
        %extracted = tensor.extract %reduced[] : tensor<i32>
        tensor.yield %extracted : i32
    } : tensor<1024x1024xi32>
    return %generated : tensor<1024x1024xi32>
}
