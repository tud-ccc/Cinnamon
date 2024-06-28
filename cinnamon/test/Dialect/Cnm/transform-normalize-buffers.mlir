// RUN: cinm-opt %s --cnm-apply-transform | FileCheck %s
#map = affine_map<(d0) -> (d0)>
module {

  // CHECK-LABEL: @simple
  // CHECK: cnm.compute
  // CHECK-NEXT: ins(%arg0[(d0, d1) -> (d0 * 512 + d1)] : memref<1024xi32>)
  // CHECK-NEXT: outs(%arg1[(d0, d1) -> (d0 * 512 + d1)] : memref<1024xi32>)
  // CHECK-NEXT: on hierarchy<2x512>
  func.func @simple(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) {
    cnm.compute
       ins(%arg0[(i, j) -> ()]: memref<1024xi32>)
       outs(%arg1[(i, j) -> (i * 512 + j)]: memref<1024xi32>)
       on hierarchy<2x512>
       do (%a1: memref<1024xi32>, %o1: memref<i32>)  {
        affine.for %i = 0 to 1024 {
          %0 = memref.load %a1[%i] : memref<1024xi32>
          %1 = memref.load %o1[] : memref<i32>
          %2 = arith.addi %0, %1 : i32
          memref.store %2, %o1[] : memref<i32>
        }
      }
    
    // into

    %r = memref.expand_shape %arg1[[0, 1]] : memref<1024xi32> into memref<2x512xi32>
    cnm.compute
       ins(%arg0[(i, j) -> ()]: memref<1024xi32>)
       outs(%r[(i, j) -> (i, j)]: memref<2x512xi32>)
       on hierarchy<2x512>
       do (%a1: memref<1024xi32>, %o1: memref<i32>)  {
        affine.for %i = 0 to 1024 {
          %0 = memref.load %a1[%i] : memref<1024xi32>
          %1 = memref.load %o1[] : memref<i32>
          %2 = arith.addi %0, %1 : i32
          memref.store %2, %o1[] : memref<i32>
        }
      }

    return
  }

  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.op<"cnm.compute">):
    transform.cnm.expand_dim %arg0 on 0 by factor 2: (!transform.op<"cnm.compute">) -> ()
  }
}