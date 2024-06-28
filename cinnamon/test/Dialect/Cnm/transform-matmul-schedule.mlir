// RUN: cinm-opt %s --cnm-apply-transform | FileCheck %s
#map = affine_map<(d0) -> (d0)>
module {


   func.func @matmul(%arg0: memref<1024x64xi32>, %arg1: memref<64x1024xi32>) {
    %arg1t = memref.alloc() : memref<1024x64xi32>
    linalg.transpose ins(%arg1: memref<64x1024xi32>) outs(%arg1t: memref<1024x64xi32>) permutation = [1,0]
    %res = memref.alloc() : memref<1024x1024xi32>

    // this is naive matmul
    cnm.compute
       ins(%arg0[(i, j) -> (i)]: memref<1024x64xi32>, 
           %arg1t[(i, j) -> (j)]: memref<1024x64xi32>)
       outs(%res[(i, j) -> (i, j)]: memref<1024x1024xi32>)
       on hierarchy<1024x1024>
       do (%a1: memref<64xi32>, %b1: memref<64xi32>, %o: memref<i32>)  {
        affine.for %i = 0 to 64 {
          %0 = affine.load %a1[%i] : memref<64xi32>
          %1 = affine.load %a1[%i] : memref<64xi32>
          %2 = arith.muli %0, %1 : i32
          %3 = affine.load %o[] : memref<i32>
          %4 = arith.addi %2, %3 : i32
          affine.store %4, %o[] : memref<i32>
        }
      }
    
    // expand dim 1 factor 16  -> <1024x16x64>
    // expand dim 0 factor 16  -> <16x64x16x64>
    // swap dim 0 and 3        -> <64x64x16x16>

    cnm.compute
      ins(%arg0[(d0, d1, d2, d3) -> (d3 * 64 + d1)] : memref<1024x64xi32>, 
          %arg1t[(d0, d1, d2, d3) -> (d2 * 64 + d0)] : memref<1024x64xi32>)
      outs(%res[(d0, d1, d2, d3) -> (d3 * 64 + d1, d2 * 64 + d0)] : memref<1024x1024xi32>) 
      on hierarchy<64x64x16x16>
      do (%arg2: memref<64xi32>, %arg3: memref<64xi32>, %arg4: memref<i32>) {
        affine.for %arg5 = 0 to 64 {
          %0 = affine.load %arg2[%arg5] : memref<64xi32>
          %1 = affine.load %arg2[%arg5] : memref<64xi32>
          %2 = arith.muli %0, %1 : i32
          %3 = affine.load %arg4[] : memref<i32>
          %4 = arith.addi %2, %3 : i32
          affine.store %4, %arg4[] : memref<i32>
        }
      }
    
    // into

    return
  }

  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.op<"cnm.compute">):
    transform.cnm.expand_dim %arg0 dim 1 by factor 16: (!transform.op<"cnm.compute">) -> ()
    transform.cnm.expand_dim %arg0 dim 0 by factor 16: (!transform.op<"cnm.compute">) -> ()
    transform.cnm.swap_dims %arg0, 0, 3: (!transform.op<"cnm.compute">) -> ()
  }
}