// RUN: cinm-opt %s | cinm-opt | FileCheck %s
// RUN: cinm-opt %s --mlir-print-op-generic | cinm-opt | FileCheck %s
module {
  // CHECK-LABEL: gemm
  func.func @gemm(%arg0: memref<8x1024xi32>, %arg1: memref<1024x128xi32>, %bias: memref<8x128xi32>) -> memref<8x128xi32> {
    %0 = cinm.compute_block (%a0 = %arg0 : memref<8x1024xi32>, %a1 = %arg1 : memref<1024x128xi32>, %bias0 = %bias : memref<8x128xi32>) -> memref<8x128xi32> attributes {workgroupShape = array<i64: 8, 128, 1>} {
      %alloc = memref.alloc() : memref<8x128xi32>
      %c0_i32 = arith.constant 0 : i32
      linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<8x128xi32>)
      cinm.op.gemm %a0, %a1 into %alloc : memref<8x1024xi32>, memref<1024x128xi32> into memref<8x128xi32>
      cinm.op.gemm %a0, %a1 plus %bias0 into %alloc : memref<8x1024xi32>, memref<1024x128xi32> plus memref<8x128xi32> into memref<8x128xi32>
      cinm.yield %alloc : memref<8x128xi32>
    }
    return %0 : memref<8x128xi32>
  }
  // CHECK-LABEL: gemv
  func.func @gemv(%arg0: memref<8x1024xi32>, %arg1: memref<1024xi32>) -> memref<8xi32> {
    %0 = cinm.compute_block (%a0 = %arg0 : memref<8x1024xi32>, %a1 = %arg1 : memref<1024xi32>) -> memref<8xi32> attributes {workgroupShape = array<i64: 1, 8, 1>} {
      %alloc = memref.alloc() : memref<8xi32>
      %c0_i32 = arith.constant 0 : i32
      linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<8xi32>)
      cinm.op.gemv %a0, %a1 into %alloc : memref<8x1024xi32>, memref<1024xi32> into memref<8xi32>
      cinm.yield %alloc : memref<8xi32>
    }
    return %0 : memref<8xi32>
  }
  // CHECK-LABEL: reduce
  func.func @reduce(%arg0: memref<8x1024xi32>, %arg1: memref<8xi32>, %arg2: memref<1024xi32>, %arg3: memref<i32>, %arg4: memref<1024xi32>) {
    // CHECK: cinm.op.reduce add(%{{.*}}) into %{{.*}} : memref<8x1024xi32> into memref<8xi32>
    cinm.op.reduce add (%arg0) into %arg1 : memref<8x1024xi32> into memref<8xi32>
    // CHECK: cinm.op.reduce maxsi(%{{.*}}) dim 0 into %{{.*}} : memref<8x1024xi32> into memref<1024xi32>
    cinm.op.reduce maxsi (%arg0) dim 0 into %arg2 : memref<8x1024xi32> into memref<1024xi32>
    // A fully-reduced destination is rank 0, not a scalar: memrefs don't
    // rank-reduce the way the tensor mode's result does.
    // CHECK: cinm.op.reduce add(%{{.*}}) into %{{.*}} : memref<1024xi32> into memref<i32>
    cinm.op.reduce add (%arg4) into %arg3 : memref<1024xi32> into memref<i32>
    return
  }
  // CHECK-LABEL: eltwise
  func.func @eltwise(%arg0: memref<6x6xi32>, %arg1: memref<6xf32>, %arg2: memref<6xf32>) {
    %alloc = memref.alloc() : memref<6x6xi32>
    cinm.op.elementwise add %arg0, %arg0 into %alloc : memref<6x6xi32> into memref<6x6xi32>
    return
  }
  // CHECK-LABEL: eltwise_into
  func.func @eltwise_into(%arg0: memref<6x6xi32>, %arg1: memref<6xf32>, %arg2: memref<6xf32>) {
    cinm.op.elementwise mul %arg1, %arg1 into %arg2 : memref<6xf32> into memref<6xf32>
    return
  }
}
