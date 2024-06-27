#map = affine_map<(d0) -> (d0)>
module {
  func.func @transform(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) {
    cnm.compute
      ins(%arg0[#map] : memref<1024xi32>)
      outs(%arg1[#map] : memref<1024xi32>) 
      on hierarchy<1024>
      do (%arg2: memref<i32>, %arg3: memref<i32>) {
        %0 = memref.load %arg2[] : memref<i32>
        %c2_i32 = arith.constant 2 : i32
        %1 = arith.muli %0, %c2_i32 : i32
        memref.store %1, %arg3[] : memref<i32>
        cnm.terminator
      }
    return
  }
  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.op<"cnm.compute">):
    transform.cnm.expand_dim %arg0 on 0 by factor 2 : (!transform.op<"cnm.compute">) -> ()
  }
}