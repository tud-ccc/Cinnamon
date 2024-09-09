#map = affine_map<(d0) -> (d0)>
module {
  func.func @transform(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) {
    cnm.compute
      ins(%arg0[#map] : memref<1024xi32>)
      outs(%arg1[#map] : memref<1024xi32>) 
      on hierarchy<1024>
      do (%arg2: memref<i32>, %arg3: memref<i32>) {
        %0 = affine.load %arg2[] : memref<i32>
        %c2_i32 = arith.constant 2 : i32
        %1 = arith.muli %0, %c2_i32 : i32
        affine.store %1, %arg3[] : memref<i32>
      }

     cnm.compute
       ins(%arg0[(i, j) -> ()]: memref<1024xi32>)
       outs(%arg1[(i, j) -> (i * 512 + j)]: memref<1024xi32>)
       on hierarchy<2x512>
       do (%a1: memref<1024xi32>, %o1: memref<i32>)  {
        affine.for %i = 0 to 1024 {
          %0 = affine.load %a1[%i] : memref<1024xi32>
          %1 = affine.load %o1[] : memref<i32>
          %2 = arith.addi %0, %1 : i32
          affine.store %2, %o1[] : memref<i32>
        }
      }
    return
  }
  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.op<"cnm.compute">):
    transform.cnm.expand_dim %arg0 dim 0 by factor 2 : (!transform.op<"cnm.compute">) -> ()
  }
}