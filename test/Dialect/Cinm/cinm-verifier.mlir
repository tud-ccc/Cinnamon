// RUN: cinm-opt --split-input-file %s --verify-diagnostics


// CHECK-LABEL: elementwise_valid_and_invalid
func.func @elementwise_valid_and_invalid(%t10 : tensor<6xf32>, %m00: memref<6xf32>) {
    cinm.op.elementwise mul %t10, %t10 into %m00 {cinm.tile_sizes = array<i64: 3>}: tensor<6xf32> into memref<6xf32>
    // expected-error @+1 {{Attribute cinm.tile_sizes tiling factor #0 (4) does not divide dimension size 6}}
    cinm.op.elementwise mul %t10, %t10 into %m00 {cinm.tile_sizes = array<i64: 4>}: tensor<6xf32> into memref<6xf32>
    return
}

// -----

// Wrong number of tiling factors for elementwise (expects 1).
func.func @elementwise_wrong_count(%t10 : tensor<6xf32>, %m00: memref<6xf32>) {
    // expected-error @+1 {{Attribute cinm.tile_sizes has 2 tiling factor(s) but op has 1 tileable dimension(s)}}
    cinm.op.elementwise mul %t10, %t10 into %m00 {cinm.tile_sizes = array<i64: 3, 3>}: tensor<6xf32> into memref<6xf32>
    return
}

// -----

// GemvOp: 2 tiling factors [tM, tK]. M=64, K=256.
func.func @gemv_valid_and_invalid(%A: memref<64x256xi32>, %x: memref<256xi32>, %out: memref<64xi32>) {
    cinm.op.gemv %A, %x into %out {cinm.tile_sizes = array<i64: 8, 32>} : memref<64x256xi32>, memref<256xi32> into memref<64xi32>
    // expected-error @+1 {{Attribute cinm.tile_sizes tiling factor #1 (48) does not divide dimension size 256}}
    cinm.op.gemv %A, %x into %out {cinm.tile_sizes = array<i64: 8, 48>} : memref<64x256xi32>, memref<256xi32> into memref<64xi32>
    return
}

// -----

// Wrong number of tiling factors for gemv (expects 2).
func.func @gemv_wrong_count(%A: memref<64x256xi32>, %x: memref<256xi32>, %out: memref<64xi32>) {
    // expected-error @+1 {{Attribute cinm.tile_sizes has 3 tiling factor(s) but op has 2 tileable dimension(s)}}
    cinm.op.gemv %A, %x into %out {cinm.tile_sizes = array<i64: 8, 32, 16>} : memref<64x256xi32>, memref<256xi32> into memref<64xi32>
    return
}

// -----

// GemmOp: 3 tiling factors [tM, tN, tK]. M=8, N=128, K=1024.
func.func @gemm_valid_and_invalid(%A: memref<8x1024xi32>, %B: memref<1024x128xi32>, %out: memref<8x128xi32>) {
    cinm.op.gemm %A, %B into %out {cinm.tile_sizes = array<i64: 4, 8, 128>} : memref<8x1024xi32>, memref<1024x128xi32> into memref<8x128xi32>
    // expected-error @+1 {{Attribute cinm.tile_sizes tiling factor #0 (3) does not divide dimension size 8}}
    cinm.op.gemm %A, %B into %out {cinm.tile_sizes = array<i64: 3, 8, 128>} : memref<8x1024xi32>, memref<1024x128xi32> into memref<8x128xi32>
    return
}

// -----

// Wrong number of tiling factors for gemm (expects 3).
func.func @gemm_wrong_count(%A: memref<8x1024xi32>, %B: memref<1024x128xi32>, %out: memref<8x128xi32>) {
    // expected-error @+1 {{Attribute cinm.tile_sizes has 2 tiling factor(s) but op has 3 tileable dimension(s)}}
    cinm.op.gemm %A, %B into %out {cinm.tile_sizes = array<i64: 4, 8>} : memref<8x1024xi32>, memref<1024x128xi32> into memref<8x128xi32>
    return
}

// -----

// ReduceOp: 1 tiling factor for the reduction dimension. tensor<12xf32>, dim=-1 (size 12).
func.func @reduce_valid_and_invalid(%v: tensor<12xf32>) {
    %r0 = cinm.op.reduce add (%v) {cinm.tile_sizes = array<i64: 4>} : tensor<12xf32> -> f32
    // expected-error @+1 {{Attribute cinm.tile_sizes tiling factor #0 (5) does not divide dimension size 12}}
    %r1 = cinm.op.reduce add (%v) {cinm.tile_sizes = array<i64: 5>} : tensor<12xf32> -> f32
    return
}

// -----

// Wrong number of tiling factors for reduce (expects 1).
func.func @reduce_wrong_count(%v: tensor<12xf32>) {
    // expected-error @+1 {{Attribute cinm.tile_sizes has 2 tiling factor(s) but op has 1 tileable dimension(s)}}
    %r = cinm.op.reduce add (%v) {cinm.tile_sizes = array<i64: 4, 4>} : tensor<12xf32> -> f32
    return
}
