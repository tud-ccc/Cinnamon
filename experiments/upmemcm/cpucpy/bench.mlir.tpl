module {
  func.func @bench(%arg0: memref<{{M}}xi32>, %arg1: memref<{{M}}xi32>) -> i32 {
    %c0_i32 = arith.constant 0 : i32
    affine.for %i = 0 to {{M}} step {{2 * T}} {
      %s0 = memref.subview %arg0[%i] [{{T}}] [1] : memref<{{M}}xi32> to memref<{{T}}xi32, strided<[1], offset: ?>>
      %s1 = memref.subview %arg1[%i] [{{T}}] [1] : memref<{{M}}xi32> to memref<{{T}}xi32, strided<[1], offset: ?>>
      memref.copy %s0, %s1 : memref<{{T}}xi32, strided<[1], offset: ?>> to memref<{{T}}xi32, strided<[1], offset: ?>>
    }
    return %c0_i32 : i32
  }
}
