// RUN: cinm-opt --convert-cinm-to-cnm %s | cinm-opt | FileCheck %s
module {
  func.func @mm_dimm8_nopt(%arg0: memref<8x1024xi32>, %arg1: memref<1024x128xi32>) -> memref<8x128xi32> {
    %0 = cinm.compute attributes {workgroupShape = array<i64: 8, 128, 1>} -> memref<8x128xi32> {
      %alloc = memref.alloc() : memref<8x128xi32>
      %c0_i32 = arith.constant 0 : i32
      linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<8x128xi32>)
      cinm.op.gemm %arg0, %arg1 into %alloc : memref<8x1024xi32>, memref<1024x128xi32> into memref<8x128xi32>
      cinm.yield %alloc : memref<8x128xi32>
    }
    return %0 : memref<8x128xi32>
  }
  func.func @gemv(%arg0: memref<8x1024xi32>, %arg1: memref<1024xi32>) -> memref<8xi32> {
    %0 = cinm.compute attributes {tileSizes = array<i64: 1, 8>, workgroupShape = array<i64: 2, 4, 1>} -> memref<8xi32> {
      %alloc = memref.alloc() : memref<8xi32>
      %c0_i32 = arith.constant 0 : i32
      linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<8xi32>)
      cinm.op.gemv %arg0, %arg1 into %alloc : memref<8x1024xi32>, memref<1024xi32> into memref<8xi32>
      cinm.yield %alloc : memref<8xi32>
    }
    return %0 : memref<8xi32>
  }
}

