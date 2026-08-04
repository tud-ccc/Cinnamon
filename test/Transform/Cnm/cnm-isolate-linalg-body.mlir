// RUN: cinm-opt %s --split-input-file --cnm-isolate-linalg-captures | FileCheck %s


// CHECK-LABEL: @capture_scalar
#map = affine_map<(d0) -> (d0)>
func.func @capture_scalar(%x: tensor<8xi32>, %c: i32) -> tensor<8xi32> {
  %e = tensor.empty() : tensor<8xi32>
// CHECK: ins(%{{.*}}, %{{.*}}: tensor<8xi32>, i32)
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
    ins(%x : tensor<8xi32>) outs(%e : tensor<8xi32>) {
  ^bb0(%in: i32, %out: i32):
    %m = arith.muli %in, %c : i32
    linalg.yield %m : i32
  } -> tensor<8xi32>
  return %r : tensor<8xi32>
}

// -----

// CHECK-LABEL: @shaped_capture
#map = affine_map<(d0) -> (d0)>
func.func @shaped_capture(%x: tensor<8xi32>, %y: tensor<8xi32>) -> tensor<8xi32> {
  %e = tensor.empty() : tensor<8xi32>
  %c0 = arith.constant 0 : index
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
    ins(%x : tensor<8xi32>) outs(%e : tensor<8xi32>) {
  ^bb0(%in: i32, %out: i32):
    %z = tensor.extract %y[%c0] : tensor<8xi32>
    linalg.yield %z : i32
  } -> tensor<8xi32>
  return %r : tensor<8xi32>
}