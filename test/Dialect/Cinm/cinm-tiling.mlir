// RUN: cinm-opt %s --cinm-isolate-compute-blocks --cinm-tiling -split-input-file | FileCheck %s

// TODO: update reduce tiling

// func.func @max(%a: tensor<1024xi32>) -> i32 {
// 	%res = cinm.compute_block (%a0 = %a : tensor<1024xi32>) -> i32 attributes { workgroupShape = array<i64: 4>, bufferSizesInBytes = array<i64: 1024> } {
// 		%d = cinm.op.reduce max (%a0): tensor<1024xi32> -> i32
// 		cinm.yield %d : i32
// 	}
// 	return %res: i32
// }
