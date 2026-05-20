// RUN: cinm-opt %s --cinm-tiling --split-input-file | FileCheck %s

// CHECK-LABEL: @max
// CHECK-SAME: (%[[A:.*]]: tensor<1024xi32>)
// CHECK:         %[[C0:.*]] = arith.constant -2147483648 : i32
// CHECK:         %[[RES:.*]] = affine.for %[[I:.*]] = 0 to 1024 step 8 iter_args(%[[ACC:.*]] = %[[C0]]) -> (i32) {
// CHECK-NEXT:      %[[SLICE:.*]] = tensor.extract_slice %[[A]][%[[I]]] [8] [1] : tensor<1024xi32> to tensor<8xi32>
// CHECK-NEXT:      %[[RED:.*]] = cinm.op.reduce maxsi(%[[SLICE]]) {{.*}} : tensor<8xi32> -> i32
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
// CHECK:         %[[EMPTY:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK:         affine.for %[[I:.*]] = 0 to 1024 step 8 iter_args(%[[ACC0:.*]] = %[[EMPTY]]) -> (tensor<1024xf32>) {
// CHECK:           affine.for %[[J:.*]] = 0 to 256 step 128 iter_args(%[[ACC1:.*]] = %[[ACC0]]) -> (tensor<1024xf32>) {
// CHECK-NEXT:        %[[SLICE:.*]] = tensor.extract_slice %[[A]][%[[I]], %[[J]]] [8, 128] [1, 1] : tensor<1024x256xf32> to tensor<8x128xf32>
// CHECK-NEXT:        %[[RED:.*]] = cinm.op.reduce minnumf(%[[SLICE]]) {{.*}} : tensor<8x128xf32> -> tensor<8xf32>
// CHECK-NEXT:        %[[INS:.*]] = tensor.insert_slice %[[RED]] into %[[ACC1]][%[[I]]] [8] [1] : tensor<8xf32> into tensor<1024xf32>
// CHECK-NEXT:        affine.yield %[[INS]] : tensor<1024xf32>
func.func @min(%a: tensor<1024x256xf32>) -> tensor<1024xf32> {
  %d = cinm.op.reduce minnumf (%a) {cinm.tile_sizes = array<i64: 8, 128>}: tensor<1024x256xf32> -> tensor<1024xf32>
	return %d: tensor<1024xf32>
}

// -----
// CHECK-LABEL: @mul
// CHECK-SAME: (%[[A:.*]]: tensor<1024x256xi32>)
// CHECK:         %[[EMPTY:.*]] = tensor.empty() : tensor<256xi32>
// CHECK:         affine.for %[[I:.*]] = 0 to 1024 step 8 iter_args(%[[ACC0:.*]] = %[[EMPTY]]) -> (tensor<256xi32>) {
// CHECK:           affine.for %[[J:.*]] = 0 to 256 step 128 iter_args(%[[ACC1:.*]] = %[[ACC0]]) -> (tensor<256xi32>) {
// CHECK-NEXT:        %[[SLICE:.*]] = tensor.extract_slice %[[A]][%[[I]], %[[J]]] [8, 128] [1, 1] : tensor<1024x256xi32> to tensor<8x128xi32>
// CHECK-NEXT:        %[[RED:.*]] = cinm.op.reduce mul(%[[SLICE]]) {{.*}} : tensor<8x128xi32> -> tensor<128xi32>
// CHECK-NEXT:        %[[INS:.*]] = tensor.insert_slice %[[RED]] into %[[ACC1]][%[[J]]] [128] [1] : tensor<128xi32> into tensor<256xi32>
// CHECK-NEXT:        affine.yield %[[INS]] : tensor<256xi32>
func.func @mul(%a: tensor<1024x256xi32>) -> tensor<256xi32> {
  %d = cinm.op.reduce mul (%a) {dimension = 0, cinm.tile_sizes = array<i64: 8, 128>}: tensor<1024x256xi32> -> tensor<256xi32>
	return %d: tensor<256xi32>
}

// -----
// CHECK-LABEL: @sum
// CHECK-SAME: (%[[A:.*]]: tensor<1024xi32>)
// CHECK:         %[[C0:.*]] = arith.constant 0 : i32
// CHECK:         %[[RES:.*]] = affine.for %[[I:.*]] = 0 to 1024 iter_args(%[[ACC:.*]] = %[[C0]]) -> (i32) {
// CHECK-NEXT:      %[[SLICE:.*]] = tensor.extract_slice %[[A]][%[[I]]] [1] [1] : tensor<1024xi32> to tensor<1xi32>
// CHECK-NEXT:      %[[RED:.*]] = cinm.op.reduce add(%[[SLICE]]) {{.*}} : tensor<1xi32> -> i32
// CHECK-NEXT:      %[[SUM:.*]] = arith.addi %[[ACC]], %[[RED]] : i32
// CHECK-NEXT:      affine.yield %[[SUM]] : i32
// CHECK:         return %[[RES]] : i32
func.func @sum(%a: tensor<1024xi32>) -> i32 {
  %d = cinm.op.reduce add (%a) {cinm.tile_sizes = array<i64: 1>}: tensor<1024xi32> -> i32
	return %d: i32
}
