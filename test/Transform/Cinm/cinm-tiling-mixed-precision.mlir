// RUN: cinm-opt %s --cinm-tiling -split-input-file | FileCheck %s

// Tiling threads the running accumulator through the `bias` slot, so the
// tiled ops take their result type from it: operand tiles stay narrow while
// the accumulator tile stays wide, with no extra plumbing.

// CHECK-LABEL: @gemv_tensor
// CHECK: affine.for %[[i:.*]] = 0 to 64 step 8 iter_args({{.*}}) -> (tensor<64xi32>)
// CHECK: affine.for %[[k:.*]] = 0 to 256 step 32
// CHECK: cinm.op.gemv %{{.*}}, %{{.*}} plus %{{.*}} into %{{.*}} : tensor<8x32xi8>, tensor<32xi8> plus tensor<8xi32> into tensor<8xi32> -> tensor<8xi32>
func.func @gemv_tensor(%A: tensor<64x256xi8>, %x: tensor<256xi8>) -> tensor<64xi32> {
  %y = cinm.op.gemv %A, %x {cinm.tile_sizes = array<i64: 8, 32>}
    : tensor<64x256xi8>, tensor<256xi8> -> tensor<64xi32>
  return %y : tensor<64xi32>
}

// -----

// CHECK-LABEL: @gemm_tensor
// CHECK: cinm.op.gemm %{{.*}}, %{{.*}} plus %{{.*}} into %{{.*}} : tensor<8x32xi8>, tensor<32x8xi8> plus tensor<8x8xi32> into tensor<8x8xi32> -> tensor<8x8xi32>
func.func @gemm_tensor(%A: tensor<64x256xi8>, %B: tensor<256x32xi8>) -> tensor<64x32xi32> {
  %c = cinm.op.gemm %A, %B {cinm.tile_sizes = array<i64: 8, 8, 32>}
    : tensor<64x256xi8>, tensor<256x32xi8> -> tensor<64x32xi32>
  return %c : tensor<64x32xi32>
}

// -----

// The memref variant accumulates into a wide out buffer; its subviews keep
// the operand and accumulator types apart the same way.

// CHECK-LABEL: @gemv_memref
// CHECK: cinm.op.gemv %{{.*}}, %{{.*}} into %{{.*}} : memref<8x32xi8, {{.*}}>, memref<32xi8, {{.*}}> into memref<8xi32, {{.*}}>
func.func @gemv_memref(%A: memref<64x256xi8>, %x: memref<256xi8>) -> memref<64xi32> {
  %o = memref.alloc() : memref<64xi32>
  %z = arith.constant 0 : i32
  linalg.fill ins(%z : i32) outs(%o : memref<64xi32>)
  cinm.op.gemv %A, %x into %o {cinm.tile_sizes = array<i64: 8, 32>}
    : memref<64x256xi8>, memref<256xi8> into memref<64xi32>
  return %o : memref<64xi32>
}
