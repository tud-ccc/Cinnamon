// RUN: cinm-opt %s --cnm-apply-transform | FileCheck %s
#map = affine_map<(d0) -> (d0)>
module {
  // CHECK-LABEL: @simple
  // CHECK: cnm.compute
  // CHECK-NEXT: ins(%arg0[(d0) -> ()] : memref<1024xi32>)
  // CHECK-NEXT: outs(%{{.*}}[(d0) -> (d0)] : memref<2x512xi32>)
  // CHECK-NEXT: on hierarchy<2>
  // CHECK-NEXT: do (%[[A:.*]]: memref<1024xi32>, %[[B:.*]]: memref<512xi32>) {
  // CHECK-NEXT:   affine.parallel (%[[i:.*]]) = (0) to (512) {
  // CHECK-NEXT:     affine.for %[[k:.*]] = 0 to 1024 {
  // CHECK-NEXT:       affine.load %[[A]][%[[k]]]
  // CHECK-NEXT:       affine.load %[[B]][%[[i]]]
  // CHECK-NEXT:       arith.addi
  // CHECK-NEXT:       affine.store %{{.*}}, %[[B]][%[[i]]]
  // CHECK-NEXT:     }
  // CHECK-NEXT:  }
  // CHECK-NEXT: }
  func.func @simple(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) {
    
    %r = memref.expand_shape %arg1[[0, 1]] : memref<1024xi32> into memref<2x512xi32>
    cnm.compute
       ins(%arg0[(i, j) -> ()]: memref<1024xi32>)
       outs(%r[(i, j) -> (i, j)]: memref<2x512xi32>)
       on hierarchy<2x512>
       do (%a1: memref<1024xi32>, %o1: memref<i32>)  {
        affine.for %i = 0 to 1024 {
          %0 = affine.load %a1[%i] : memref<1024xi32>
          %1 = affine.load %o1[] : memref<i32>
          %2 = arith.addi %0, %1 : i32
          affine.store %2, %o1[] : memref<i32>
        }
      }


    // cnm.compute
    //    ins(%arg0[(i) -> ()]: memref<1024xi32>)
    //    outs(%r[(i) -> (i)]: memref<2x512xi32>)
    //    on hierarchy<2>
    //    do (%a1: memref<1024xi32>, %o1: memref<512xi32>)  {
    //     affine.parallel (%j) = (0) to (512) {
    //         affine.for %i = 0 to 1024 {
    //           %0 = affine.load %a1[%i] : memref<1024xi32>
    //           %1 = affine.load %o1[%j] : memref<512xi32>
    //           %2 = arith.addi %0, %1 : i32
    //           affine.store %2, %o1[%j] : memref<512xi32>
    //         }
    //     }
    //   }

    return
  }
  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.op<"cnm.compute">):
    transform.cnm.peel_right %arg0: (!transform.op<"cnm.compute">) -> ()
  }
}