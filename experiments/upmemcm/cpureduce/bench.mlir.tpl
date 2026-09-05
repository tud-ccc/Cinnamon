module {
  func.func @bench(%arg0: memref<{{M}}xi32>) -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %x = affine.for %i = 0 to {{M}} iter_args(%acc = %c0_i32) -> i32 {
      %0 = memref.load %arg0[%i] : memref<{{M}}xi32>
      %1 = arith.addi %0, %acc : i32
      affine.yield %1: i32
    }
    return %x : i32
  }
}
