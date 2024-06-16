#map = affine_map<(d0, d1, d2) -> (d0 * 128 + d1 + d2)>
module {
  memref.global "private" constant @__constant_1xi64_1 : memref<1xi64> = dense<32768>
  memref.global "private" constant @__constant_2048x16xi32 : memref<2048x16xi32> = dense<0>
  memref.global "private" constant @__constant_2xi64_2 : memref<2xi64> = dense<[2048, 16]>
  memref.global "private" constant @__constant_32768xi32 : memref<32768xi32> = dense<0>
  memref.global "private" constant @__constant_2xi64_1 : memref<2xi64> = dense<[16, 1048576]>
  memref.global "private" constant @__constant_1xi64_0 : memref<1xi64> = dense<16384>
  memref.global "private" constant @__constant_1024x16xi32 : memref<1024x16xi32> = dense<0>
  memref.global "private" constant @__constant_2xi64_0 : memref<2xi64> = dense<[1024, 16]>
  memref.global "private" constant @__constant_16384xi32 : memref<16384xi32> = dense<0>
  memref.global "private" constant @__constant_1xi64 : memref<1xi64> = dense<16777216>
  memref.global "private" constant @__constant_2xi64 : memref<2xi64> = dense<[8, 2097152]>
  func.func @va_8(%arg0: memref<8x2097152xi32>, %arg1: memref<8x2097152xi32>) {
    %0 = memref.get_global @__constant_2xi64 : memref<2xi64>
    %1 = memref.get_global @__constant_1xi64 : memref<1xi64>
    %reshape = memref.reshape %arg0(%1) : (memref<8x2097152xi32>, memref<1xi64>) -> memref<16777216xi32>
    %reshape_0 = memref.reshape %arg1(%1) : (memref<8x2097152xi32>, memref<1xi64>) -> memref<16777216xi32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<16777216xi32>
    %c0 = arith.constant 0 : index
    %c16777216 = arith.constant 16777216 : index
    %c16384 = arith.constant 16384 : index
    %2 = scf.for %arg2 = %c0 to %c16777216 step %c16384 iter_args(%arg3 = %alloc) -> (memref<16777216xi32>) {
      %subview = memref.subview %reshape[%arg2] [16384] [1] : memref<16777216xi32> to memref<16384xi32, strided<[1], offset: ?>>
      %subview_1 = memref.subview %reshape_0[%arg2] [16384] [1] : memref<16777216xi32> to memref<16384xi32, strided<[1], offset: ?>>
      %3 = cnm.workgroup : !cnm.workgroup<8x128x1>
      %4 = memref.get_global @__constant_2xi64_0 : memref<2xi64>
      %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<16384xi32>
      memref.copy %subview, %alloc_2 : memref<16384xi32, strided<[1], offset: ?>> to memref<16384xi32>
      %reshape_3 = memref.reshape %alloc_2(%4) : (memref<16384xi32>, memref<2xi64>) -> memref<1024x16xi32>
      %5 = bufferization.to_tensor %reshape_3 : memref<1024x16xi32>
      %6 = cnm.alloc() for %3 : !cnm.buffer<16xi32 on 8x128x1, level 0>
      %7 = cnm.scatter %5 into %6[#map] of %3 : tensor<1024x16xi32> into !cnm.buffer<16xi32 on 8x128x1, level 0>
      %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<16384xi32>
      memref.copy %subview_1, %alloc_4 : memref<16384xi32, strided<[1], offset: ?>> to memref<16384xi32>
      %reshape_5 = memref.reshape %alloc_4(%4) : (memref<16384xi32>, memref<2xi64>) -> memref<1024x16xi32>
      %8 = bufferization.to_tensor %reshape_5 : memref<1024x16xi32>
      %9 = cnm.alloc() for %3 : !cnm.buffer<16xi32 on 8x128x1, level 0>
      %10 = cnm.scatter %8 into %9[#map] of %3 : tensor<1024x16xi32> into !cnm.buffer<16xi32 on 8x128x1, level 0>
      %11 = memref.get_global @__constant_1024x16xi32 : memref<1024x16xi32>
      %12 = bufferization.to_tensor %11 : memref<1024x16xi32>
      %13 = cnm.alloc() for %3 : !cnm.buffer<16xi32 on 8x128x1, level 0>
      %14 = cnm.scatter %12 into %13[#map] of %3 : tensor<1024x16xi32> into !cnm.buffer<16xi32 on 8x128x1, level 0>
      %15 = cnm.launch %3 in(%6, %9 : !cnm.buffer<16xi32 on 8x128x1, level 0>, !cnm.buffer<16xi32 on 8x128x1, level 0>) out(%13 : !cnm.buffer<16xi32 on 8x128x1, level 0>) on !cnm.workgroup<8x128x1> {
      ^bb0(%arg4: memref<16xi32>, %arg5: memref<16xi32>, %arg6: memref<16xi32>):
        linalg.add ins(%arg4, %arg5 : memref<16xi32>, memref<16xi32>) outs(%arg6 : memref<16xi32>)
      }
      %output, %token = cnm.gather %13[#map] of %3 : !cnm.buffer<16xi32 on 8x128x1, level 0> into tensor<1024x16xi32>
      %16 = bufferization.to_memref %output : memref<1024x16xi32>
      %17 = memref.get_global @__constant_1xi64_0 : memref<1xi64>
      %reshape_6 = memref.reshape %16(%17) : (memref<1024x16xi32>, memref<1xi64>) -> memref<16384xi32>
      %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<16777216xi32>
      memref.copy %arg3, %alloc_7 : memref<16777216xi32> to memref<16777216xi32>
      %subview_8 = memref.subview %alloc_7[%arg2] [16384] [1] : memref<16777216xi32> to memref<16384xi32, strided<[1], offset: ?>>
      memref.copy %reshape_6, %subview_8 : memref<16384xi32> to memref<16384xi32, strided<[1], offset: ?>>
      scf.yield %alloc_7 : memref<16777216xi32>
    }
    return
  }
  func.func @va_16(%arg0: memref<16x1048576xi32>, %arg1: memref<16x1048576xi32>) {
    %0 = memref.get_global @__constant_2xi64_1 : memref<2xi64>
    %1 = memref.get_global @__constant_1xi64 : memref<1xi64>
    %reshape = memref.reshape %arg0(%1) : (memref<16x1048576xi32>, memref<1xi64>) -> memref<16777216xi32>
    %reshape_0 = memref.reshape %arg1(%1) : (memref<16x1048576xi32>, memref<1xi64>) -> memref<16777216xi32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<16777216xi32>
    %c0 = arith.constant 0 : index
    %c16777216 = arith.constant 16777216 : index
    %c32768 = arith.constant 32768 : index
    %2 = scf.for %arg2 = %c0 to %c16777216 step %c32768 iter_args(%arg3 = %alloc) -> (memref<16777216xi32>) {
      %subview = memref.subview %reshape[%arg2] [32768] [1] : memref<16777216xi32> to memref<32768xi32, strided<[1], offset: ?>>
      %subview_1 = memref.subview %reshape_0[%arg2] [32768] [1] : memref<16777216xi32> to memref<32768xi32, strided<[1], offset: ?>>
      %3 = cnm.workgroup : !cnm.workgroup<16x128x1>
      %4 = memref.get_global @__constant_2xi64_2 : memref<2xi64>
      %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<32768xi32>
      memref.copy %subview, %alloc_2 : memref<32768xi32, strided<[1], offset: ?>> to memref<32768xi32>
      %reshape_3 = memref.reshape %alloc_2(%4) : (memref<32768xi32>, memref<2xi64>) -> memref<2048x16xi32>
      %5 = bufferization.to_tensor %reshape_3 : memref<2048x16xi32>
      %6 = cnm.alloc() for %3 : !cnm.buffer<16xi32 on 16x128x1, level 0>
      %7 = cnm.scatter %5 into %6[#map] of %3 : tensor<2048x16xi32> into !cnm.buffer<16xi32 on 16x128x1, level 0>
      %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<32768xi32>
      memref.copy %subview_1, %alloc_4 : memref<32768xi32, strided<[1], offset: ?>> to memref<32768xi32>
      %reshape_5 = memref.reshape %alloc_4(%4) : (memref<32768xi32>, memref<2xi64>) -> memref<2048x16xi32>
      %8 = bufferization.to_tensor %reshape_5 : memref<2048x16xi32>
      %9 = cnm.alloc() for %3 : !cnm.buffer<16xi32 on 16x128x1, level 0>
      %10 = cnm.scatter %8 into %9[#map] of %3 : tensor<2048x16xi32> into !cnm.buffer<16xi32 on 16x128x1, level 0>
      %11 = memref.get_global @__constant_2048x16xi32 : memref<2048x16xi32>
      %12 = bufferization.to_tensor %11 : memref<2048x16xi32>
      %13 = cnm.alloc() for %3 : !cnm.buffer<16xi32 on 16x128x1, level 0>
      %14 = cnm.scatter %12 into %13[#map] of %3 : tensor<2048x16xi32> into !cnm.buffer<16xi32 on 16x128x1, level 0>
      %15 = cnm.launch %3 in(%6, %9 : !cnm.buffer<16xi32 on 16x128x1, level 0>, !cnm.buffer<16xi32 on 16x128x1, level 0>) out(%13 : !cnm.buffer<16xi32 on 16x128x1, level 0>) on !cnm.workgroup<16x128x1> {
      ^bb0(%arg4: memref<16xi32>, %arg5: memref<16xi32>, %arg6: memref<16xi32>):
        linalg.add ins(%arg4, %arg5 : memref<16xi32>, memref<16xi32>) outs(%arg6 : memref<16xi32>)
      }
      %output, %token = cnm.gather %13[#map] of %3 : !cnm.buffer<16xi32 on 16x128x1, level 0> into tensor<2048x16xi32>
      %16 = bufferization.to_memref %output : memref<2048x16xi32>
      %17 = memref.get_global @__constant_1xi64_1 : memref<1xi64>
      %reshape_6 = memref.reshape %16(%17) : (memref<2048x16xi32>, memref<1xi64>) -> memref<32768xi32>
      %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<16777216xi32>
      memref.copy %arg3, %alloc_7 : memref<16777216xi32> to memref<16777216xi32>
      %subview_8 = memref.subview %alloc_7[%arg2] [32768] [1] : memref<16777216xi32> to memref<32768xi32, strided<[1], offset: ?>>
      memref.copy %reshape_6, %subview_8 : memref<32768xi32> to memref<32768xi32, strided<[1], offset: ?>>
      scf.yield %alloc_7 : memref<16777216xi32>
    }
    return
  }
}

