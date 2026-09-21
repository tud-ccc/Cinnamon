// RUN: cinm-opt %s --cinm-merge-host-blocks --split-input-file | FileCheck %s

// Two host blocks separated only by views (reshape, assembling into a
// buffer that starts as tensor.empty) become one. The views move inside,
// the tensor.empty stays above, and the values used afterwards come out.

// CHECK-LABEL: func.func @views_between
// CHECK:         %[[E:.*]] = tensor.empty() : tensor<16xi32>
// CHECK:         %[[M:.*]]:2 = cinm.compute -> tensor<16xi32>, tensor<8xi32> attributes {cinm.available_platforms = [#cinm.host_platform]}
// CHECK-NEXT:      arith.addi
// CHECK-NEXT:      tensor.collapse_shape
// CHECK-NEXT:      tensor.insert_slice {{.*}} into %[[E]]
// CHECK-NEXT:      tensor.extract_slice
// CHECK-NEXT:      arith.muli
// CHECK-NEXT:      cinm.yield
// CHECK-NOT:     cinm.compute
// CHECK:         return %[[M]]#0, %[[M]]#1
func.func @views_between(%x: tensor<1x8xi32>) -> (tensor<16xi32>, tensor<8xi32>) {
  %a = cinm.compute -> tensor<1x8xi32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
    %s = arith.addi %x, %x : tensor<1x8xi32>
    cinm.yield %s : tensor<1x8xi32>
  }
  %c = tensor.collapse_shape %a [[0, 1]] : tensor<1x8xi32> into tensor<8xi32>
  %e = tensor.empty() : tensor<16xi32>
  %i = tensor.insert_slice %c into %e[0] [8] [1] : tensor<8xi32> into tensor<16xi32>
  %v = tensor.extract_slice %i[0] [8] [1] : tensor<16xi32> to tensor<8xi32>
  %b = cinm.compute -> tensor<8xi32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
    %m = arith.muli %v, %v : tensor<8xi32>
    cinm.yield %m : tensor<8xi32>
  }
  return %i, %b : tensor<16xi32>, tensor<8xi32>
}

// -----

// A device block between them keeps them apart.

// CHECK-LABEL: func.func @device_between
// CHECK-COUNT-3: cinm.compute
// CHECK-NOT:     cinm.compute
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>
func.func @device_between(%w: tensor<8x8xi32>, %x: tensor<8xi32>) -> tensor<8xi32> {
  %a = cinm.compute -> tensor<8xi32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
    %s = arith.addi %x, %x : tensor<8xi32>
    cinm.yield %s : tensor<8xi32>
  }
  %g = cinm.compute -> tensor<8xi32> attributes {cinm.available_platforms = [#upmem]} {
    %r = cinm.op.gemv %w, %a : tensor<8x8xi32>, tensor<8xi32> -> tensor<8xi32>
    cinm.yield %r : tensor<8xi32>
  }
  %b = cinm.compute -> tensor<8xi32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
    %m = arith.muli %g, %g : tensor<8xi32>
    cinm.yield %m : tensor<8xi32>
  }
  return %b : tensor<8xi32>
}

// -----

// Writing into a function argument (a cache) is state: the run stops there,
// and the insert stays outside between the two blocks.

// CHECK-LABEL: func.func @state_between
// CHECK:         cinm.compute
// CHECK:         tensor.insert_slice {{.*}} into %{{.*}}[0, 0] [1, 8] [1, 1] : tensor<8xi32> into tensor<4x8xi32>
// CHECK:         cinm.compute
func.func @state_between(%x: tensor<8xi32>, %cache: tensor<4x8xi32>) -> (tensor<4x8xi32>, tensor<8xi32>) {
  %a = cinm.compute -> tensor<8xi32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
    %s = arith.addi %x, %x : tensor<8xi32>
    cinm.yield %s : tensor<8xi32>
  }
  %u = tensor.insert_slice %a into %cache[0, 0] [1, 8] [1, 1] : tensor<8xi32> into tensor<4x8xi32>
  %b = cinm.compute -> tensor<8xi32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
    %m = arith.muli %a, %a : tensor<8xi32>
    cinm.yield %m : tensor<8xi32>
  }
  return %u, %b : tensor<4x8xi32>, tensor<8xi32>
}

// -----

// Inside a loop body, the run is merged within the body.

// CHECK-LABEL: func.func @in_loop
// CHECK:         scf.for
// CHECK:           cinm.compute -> tensor<8xi32> attributes {cinm.available_platforms = [#cinm.host_platform]}
// CHECK-NOT:       cinm.compute
// CHECK:           scf.yield
func.func @in_loop(%x0: tensor<8xi32>) -> tensor<8xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%x = %x0) -> (tensor<8xi32>) {
    %a = cinm.compute -> tensor<2x4xi32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %s = arith.addi %x, %x : tensor<8xi32>
      %e = tensor.expand_shape %s [[0, 1]] output_shape [2, 4] : tensor<8xi32> into tensor<2x4xi32>
      cinm.yield %e : tensor<2x4xi32>
    }
    %c = tensor.collapse_shape %a [[0, 1]] : tensor<2x4xi32> into tensor<8xi32>
    %b = cinm.compute -> tensor<8xi32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %m = arith.muli %c, %c : tensor<8xi32>
      cinm.yield %m : tensor<8xi32>
    }
    scf.yield %b : tensor<8xi32>
  }
  return %r : tensor<8xi32>
}
