// RUN: cinm-opt %s --cinm-tiling --split-input-file | FileCheck %s

// When the result of a tile is a scalar (the whole reduction fits one tile
// result), the accumulator is a scalar and each trip combines into it with an
// arith op. When the result is shaped, the reduction dimension may still be
// split into several trips writing the same accumulator slice, so that slice
// must be read, combined and written back -- and the accumulator must start
// at the reduction's identity, since its initial contents are read.

// CHECK-LABEL: @max
// CHECK-SAME: (%[[A:.*]]: tensor<1024xi32>)
// CHECK:         %[[C0:.*]] = arith.constant -2147483648 : i32
// CHECK:         %[[RES:.*]] = affine.for %[[I:.*]] = 0 to 1024 step 8 iter_args(%[[ACC:.*]] = %[[C0]]) -> (i32) {
// CHECK-NEXT:      %[[SLICE:.*]] = tensor.extract_slice %[[A]][%[[I]]] [8] [1] : tensor<1024xi32> to tensor<8xi32>
// CHECK-NEXT:      %[[RED:.*]] = cinm.op.reduce maxsi(%[[SLICE]]) : tensor<8xi32> -> i32
// CHECK-NEXT:      %[[MAX:.*]] = arith.maxsi %[[ACC]], %[[RED]] : i32
// CHECK-NEXT:      affine.yield %[[MAX]] : i32
// CHECK:         return %[[RES]] : i32
func.func @max(%a: tensor<1024xi32>) -> i32 {
  %d = cinm.op.reduce maxsi (%a){cinm.tile_sizes = array<i64: 8>}: tensor<1024xi32> -> i32
	return %d: i32
}

// -----
// CHECK-LABEL: @min
// CHECK-SAME: (%[[A:.*]]: tensor<1024x256xf32>)
// The reduction dim (256) is split into 2 trips of 128 over the same
// accumulator slice, so the partial minima have to be combined.
// CHECK:         %[[INIT:.*]] = arith.constant dense<0x7FC00000> : tensor<1024xf32>
// CHECK:         affine.for %[[I:.*]] = 0 to 1024 step 8 iter_args(%[[ACC0:.*]] = %[[INIT]]) -> (tensor<1024xf32>) {
// CHECK:           affine.for %[[J:.*]] = 0 to 256 step 128 iter_args(%[[ACC1:.*]] = %[[ACC0]]) -> (tensor<1024xf32>) {
// CHECK-NEXT:        %[[SLICE:.*]] = tensor.extract_slice %[[A]][%[[I]], %[[J]]] [8, 128] [1, 1] : tensor<1024x256xf32> to tensor<8x128xf32>
// CHECK-NEXT:        %[[RED:.*]] = cinm.op.reduce minnumf(%[[SLICE]]) : tensor<8x128xf32> -> tensor<8xf32>
// CHECK-NEXT:        %[[OLD:.*]] = tensor.extract_slice %[[ACC1]][%[[I]]] [8] [1] : tensor<1024xf32> to tensor<8xf32>
// CHECK-NEXT:        %[[NEW:.*]] = linalg.map { arith.minnumf } ins(%[[OLD]], %[[RED]] : tensor<8xf32>, tensor<8xf32>) outs(%[[OLD]] : tensor<8xf32>)
// CHECK-NEXT:        %[[INS:.*]] = tensor.insert_slice %[[NEW]] into %[[ACC1]][%[[I]]] [8] [1] : tensor<8xf32> into tensor<1024xf32>
// CHECK-NEXT:        affine.yield %[[INS]] : tensor<1024xf32>
func.func @min(%a: tensor<1024x256xf32>) -> tensor<1024xf32> {
  %d = cinm.op.reduce minnumf (%a) {cinm.tile_sizes = array<i64: 8, 128>}: tensor<1024x256xf32> -> tensor<1024xf32>
	return %d: tensor<1024xf32>
}

// -----
// CHECK-LABEL: @mul
// CHECK-SAME: (%[[A:.*]]: tensor<1024x256xi32>)
// Reduction over dim 0 (1024) tiled by 8: 128 trips over the same slice. The
// identity for `mul` is 1, not 0 -- seeding with a zeroed or undefined buffer
// would annihilate the product.
// CHECK:         %[[INIT:.*]] = arith.constant dense<1> : tensor<256xi32>
// CHECK:         affine.for %[[I:.*]] = 0 to 1024 step 8 iter_args(%[[ACC0:.*]] = %[[INIT]]) -> (tensor<256xi32>) {
// CHECK:           affine.for %[[J:.*]] = 0 to 256 step 128 iter_args(%[[ACC1:.*]] = %[[ACC0]]) -> (tensor<256xi32>) {
// CHECK-NEXT:        %[[SLICE:.*]] = tensor.extract_slice %[[A]][%[[I]], %[[J]]] [8, 128] [1, 1] : tensor<1024x256xi32> to tensor<8x128xi32>
// CHECK-NEXT:        %[[RED:.*]] = cinm.op.reduce mul(%[[SLICE]]) dim 0 : tensor<8x128xi32> -> tensor<128xi32>
// CHECK-NEXT:        %[[OLD:.*]] = tensor.extract_slice %[[ACC1]][%[[J]]] [128] [1] : tensor<256xi32> to tensor<128xi32>
// CHECK-NEXT:        %[[NEW:.*]] = linalg.map { arith.muli{{.*}} } ins(%[[OLD]], %[[RED]] : tensor<128xi32>, tensor<128xi32>) outs(%[[OLD]] : tensor<128xi32>)
// CHECK-NEXT:        %[[INS:.*]] = tensor.insert_slice %[[NEW]] into %[[ACC1]][%[[J]]] [128] [1] : tensor<128xi32> into tensor<256xi32>
// CHECK-NEXT:        affine.yield %[[INS]] : tensor<256xi32>
func.func @mul(%a: tensor<1024x256xi32>) -> tensor<256xi32> {
  %d = cinm.op.reduce mul (%a) dim 0 {cinm.tile_sizes = array<i64: 8, 128>}: tensor<1024x256xi32> -> tensor<256xi32>
	return %d: tensor<256xi32>
}

// -----
// CHECK-LABEL: @sum
// CHECK-SAME: (%[[A:.*]]: tensor<1024xi32>)
// CHECK:         %[[C0:.*]] = arith.constant 0 : i32
// CHECK:         %[[RES:.*]] = affine.for %[[I:.*]] = 0 to 1024 iter_args(%[[ACC:.*]] = %[[C0]]) -> (i32) {
// CHECK-NEXT:      %[[SLICE:.*]] = tensor.extract_slice %[[A]][%[[I]]] [1] [1] : tensor<1024xi32> to tensor<1xi32>
// CHECK-NEXT:      %[[RED:.*]] = cinm.op.reduce add(%[[SLICE]]) : tensor<1xi32> -> i32
// CHECK-NEXT:      %[[SUM:.*]] = arith.addi %[[ACC]], %[[RED]] : i32
// CHECK-NEXT:      affine.yield %[[SUM]] : i32
// CHECK:         return %[[RES]] : i32
func.func @sum(%a: tensor<1024xi32>) -> i32 {
  %d = cinm.op.reduce add (%a) {cinm.tile_sizes = array<i64: 1>}: tensor<1024xi32> -> i32
	return %d: i32
}
