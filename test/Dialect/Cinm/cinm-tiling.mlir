// RUN: cinm-opt %s --cinm-tiling --split-input-file | FileCheck %s

// TODO: update reduce tiling

// CHECK-LABEL: @max
func.func @max(%a: tensor<1024xi32>) -> i32 {
  %d = cinm.op.reduce max (%a){cinm.tile_sizes = array<i64: 8>}: tensor<1024xi32> -> i32
	return %d: i32
}

// -----
// CHECK-LABEL: @min
func.func @min(%a: tensor<1024x256xi32>) -> tensor<1024xi32> {
  %d = cinm.op.reduce max (%a) {cinm.tile_sizes = array<i64: 8, 128>}: tensor<1024x256xi32> -> tensor<1024xi32>
	return %d: tensor<1024xi32>
}

// -----
// CHECK-LABEL: @mul
func.func @mul(%a: tensor<1024x256xi32>) -> tensor<256xi32> {
  %d = cinm.op.reduce mul (%a) {dimension = 0, cinm.tile_sizes = array<i64: 8, 128>}: tensor<1024x256xi32> -> tensor<256xi32>
	return %d: tensor<256xi32>
}

// -----
// CHECK-LABEL: @sum
func.func @sum(%a: tensor<1024xi32>) -> i32 {
  %d = cinm.op.reduce add (%a) {cinm.tile_sizes = array<i64: 1>}: tensor<1024xi32> -> i32
	return %d: i32
}
