
// RUN: cinm-opt %s --cnm-apply-transform | FileCheck %s
#map = affine_map<(d0) -> (d0)>
module {


   func.func @matmul(%arg0: memref<1024x64xi32>, %arg1: memref<64x1024xi32>) {
    %arg1t = memref.alloc() : memref<1024x64xi32>
    linalg.transpose ins(%arg1: memref<64x1024xi32>) outs(%arg1t: memref<1024x64xi32>) permutation = [1,0]
    %res = memref.alloc() : memref<1024x1024xi32>

    // peel right (need a reshape first), reshape turns output map into (i,j,k,l) -> (i*16+k,j,l)
    %exp0 = memref.expand_shape %res[[0], [1, 2]] : memref<1024x1024xi32> into memref<1024x64x16xi32>
    %expb = memref.expand_shape %arg1t[[0, 1], [2]] : memref<1024x64xi32> into memref<64x16x64xi32>

    cnm.compute
      ins(%arg0[(d0, d1, d2) -> (d0 * 16 + d2)] : memref<1024x64xi32>, 
          %expb[(d0, d1, d2) -> (d1)] : memref<64x16x64xi32>)
      outs(%exp0[(d0, d1, d2) -> (d0 * 16 + d2, d1)] : memref<1024x64x16xi32>) 
      on hierarchy<64x64x16>
      do (%arg2: memref<64xi32>, %arg3: memref<16x64xi32>, %arg4: memref<16xi32>) {
        affine.parallel (%x) = (0) to (16) {
          affine.for %arg5 = 0 to 64 {
            %0 = affine.load %arg2[%arg5] : memref<64xi32>
            %1 = affine.load %arg3[%x, %arg5] : memref<16x64xi32>
            %2 = arith.muli %0, %1 : i32
            %3 = affine.load %arg4[%x] : memref<16xi32>
            %4 = arith.addi %2, %3 : i32
            affine.store %4, %arg4[%x] : memref<16xi32>
          }
        }
      }



    // affine.parallel (%D0, %D1) = (0, 0) to (64, 64) {
    //   cnm.compute 
    //     symbols [%D0, %D1]
    //     ins(%arg0[(d2, d3)[D0, D1] -> (d3 * 64 + D1)] : memref<1024x64xi32>, 
    //         %arg1t[(d2, d3)[D0, D1] -> (d2 * 64 + D0)] : memref<1024x64xi32>)
    //     outs(%res[(d2, d3)[D0, D1] -> (d3 * 64 + D1, d2 * 64 + D0)] : memref<1024x1024xi32>) 
    //     on hierarchy<16x16>
    //     do (%arg2: memref<64xi32>, %arg3: memref<64xi32>, %arg4: memref<i32>) {
    //       affine.for %arg5 = 0 to 64 {
    //         %0 = affine.load %arg2[%arg5] : memref<64xi32>
    //         %1 = affine.load %arg2[%arg5] : memref<64xi32>
    //         %2 = arith.muli %0, %1 : i32
    //         %3 = affine.load %arg4[] : memref<i32>
    //         %4 = arith.addi %2, %3 : i32
    //         affine.store %4, %arg4[] : memref<i32>
    //       }
    //     }
    // }
    
    // into

    return
  }


  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.op<"affine.parallel">):

  }
}