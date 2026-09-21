// RUN: cinm-opt %s --cinm-outline-host-blocks --split-input-file | FileCheck %s
// The split runs over the whole file as one module, so the written module
// has to hold the functions of every case.
// RUN: cinm-opt %s --cinm-outline-host-blocks=outlined-file=%t.mlir | FileCheck %s --check-prefix=SPLIT
// RUN: FileCheck %s --check-prefix=FILE < %t.mlir

// A host block between two device blocks. Its body becomes a call; the value
// it reads from above is passed, the constant and the tensor.empty it reads
// are recreated inside the function. The device blocks are left alone.

// CHECK-LABEL: func.func @bridge
// CHECK:         %[[V:.*]] = cinm.compute -> tensor<8xi32> attributes {cinm.available_platforms = [#upmem]}
// CHECK:           cinm.op.gemv
// CHECK:         %[[H:.*]] = cinm.compute -> tensor<8xi32> attributes {cinm.available_platforms = [#cinm.host_platform]}
// CHECK-NEXT:      %[[R:.*]] = func.call @bridge_host0(%[[V]]) : (tensor<8xi32>) -> tensor<8xi32>
// CHECK-NEXT:      cinm.yield %[[R]]
// CHECK:         cinm.compute -> tensor<8xi32> attributes {cinm.available_platforms = [#upmem]}
// CHECK:       module @outlined
// CHECK:         func.func @bridge_host0(%[[A:.*]]: tensor<8xi32>) -> tensor<8xi32>
// CHECK-DAG:       arith.constant 3 : i32
// CHECK-DAG:       tensor.empty() : tensor<8xi32>
// CHECK:           linalg.generic
// CHECK:           return
// CHECK:       func.func private @bridge_host0(tensor<8xi32>) -> tensor<8xi32>

// SPLIT-NOT:   module @outlined
// SPLIT-DAG:   func.call @bridge_host0
// SPLIT-DAG:   func.call @in_loop_host0
// SPLIT-DAG:   func.call @isolated_host0
// SPLIT-DAG:   func.func private @bridge_host0(tensor<8xi32>) -> tensor<8xi32>
// SPLIT-DAG:   func.func private @in_loop_host0(index, tensor<4xi32>) -> tensor<4xi32>
// SPLIT-DAG:   func.func private @isolated_host0(tensor<4xi32>) -> tensor<4xi32>

// FILE:        module @outlined
// FILE-DAG:      func.func @bridge_host0(%{{.*}}: tensor<8xi32>) -> tensor<8xi32>
// FILE-DAG:      func.func @in_loop_host0(%{{.*}}: index, %{{.*}}: tensor<4xi32>) -> tensor<4xi32>
// FILE-DAG:      func.func @isolated_host0(%{{.*}}: tensor<4xi32>) -> tensor<4xi32>
// FILE-DAG:      func.func private @double
// FILE-DAG:      func.func private @twice

#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>
#id = affine_map<(i) -> (i)>
func.func @bridge(%a: tensor<8x8xi32>, %x: tensor<8xi32>) -> tensor<8xi32> {
  %c3 = arith.constant 3 : i32
  %v = cinm.compute -> tensor<8xi32> attributes {cinm.available_platforms = [#upmem]} {
    %g = cinm.op.gemv %a, %x : tensor<8x8xi32>, tensor<8xi32> -> tensor<8xi32>
    cinm.yield %g : tensor<8xi32>
  }
  %e = tensor.empty() : tensor<8xi32>
  %h = cinm.compute -> tensor<8xi32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
    %m = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
        ins(%v : tensor<8xi32>) outs(%e : tensor<8xi32>) {
    ^bb0(%in: i32, %out: i32):
      %p = arith.muli %in, %c3 : i32
      linalg.yield %p : i32
    } -> tensor<8xi32>
    cinm.yield %m : tensor<8xi32>
  }
  %w = cinm.compute -> tensor<8xi32> attributes {cinm.available_platforms = [#upmem]} {
    %g = cinm.op.gemv %a, %h : tensor<8x8xi32>, tensor<8xi32> -> tensor<8xi32>
    cinm.yield %g : tensor<8xi32>
  }
  return %w : tensor<8xi32>
}

// -----

// A host block inside a rolled loop captures the loop's own values; the
// call stays in the loop and passes them on every iteration, in the order
// the body first uses them. Numbering is per enclosing function.

// CHECK-LABEL: func.func @in_loop
// CHECK:         scf.for %[[I:[a-z0-9]+]] = {{.*}} iter_args(%[[X:[a-z0-9]+]] = {{.*}})
// CHECK:           cinm.compute -> tensor<4xi32> attributes {cinm.available_platforms = [#cinm.host_platform]}
// CHECK-NEXT:        func.call @in_loop_host0(%[[I]], %[[X]]) : (index, tensor<4xi32>) -> tensor<4xi32>
// CHECK:       module @outlined
// CHECK:         func.func @in_loop_host0(%{{.*}}: index, %{{.*}}: tensor<4xi32>) -> tensor<4xi32>
// CHECK:       func.func private @in_loop_host0(index, tensor<4xi32>) -> tensor<4xi32>
func.func @in_loop(%x0: tensor<4xi32>) -> tensor<4xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%x = %x0) -> (tensor<4xi32>) {
    %y = cinm.compute -> tensor<4xi32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %ii = arith.index_cast %i : index to i32
      %s = tensor.insert %ii into %x[%i] : tensor<4xi32>
      cinm.yield %s : tensor<4xi32>
    }
    scf.yield %y : tensor<4xi32>
  }
  return %r : tensor<4xi32>
}

// -----

// An isolated compute_block passes its region arguments, and a function the
// body calls is copied into the outlined module (along with its own callee).

// CHECK-LABEL: func.func @isolated
// CHECK:         cinm.compute_block (%[[B:[a-z0-9]+]] = %{{[a-z0-9]+}} : tensor<4xi32>) -> tensor<4xi32> attributes {cinm.available_platforms = [#cinm.host_platform]}
// CHECK-NEXT:      %[[R:.*]] = func.call @isolated_host0(%[[B]])
// CHECK-NEXT:      cinm.yield %[[R]]
// CHECK:       module @outlined
// CHECK-DAG:     func.func @isolated_host0
// CHECK-DAG:     func.func private @double
// CHECK-DAG:     func.func private @twice
func.func private @twice(%t: tensor<4xi32>) -> tensor<4xi32> {
  %s = arith.addi %t, %t : tensor<4xi32>
  return %s : tensor<4xi32>
}
func.func private @double(%t: tensor<4xi32>) -> tensor<4xi32> {
  %s = func.call @twice(%t) : (tensor<4xi32>) -> tensor<4xi32>
  return %s : tensor<4xi32>
}
func.func @isolated(%x: tensor<4xi32>) -> tensor<4xi32> {
  %y = cinm.compute_block (%b = %x : tensor<4xi32>) -> tensor<4xi32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
    %d = func.call @double(%b) : (tensor<4xi32>) -> tensor<4xi32>
    cinm.yield %d : tensor<4xi32>
  }
  return %y : tensor<4xi32>
}

// -----

// A destination pin is dropped from the outlined copy: the function returns
// the pinned value itself.

// CHECK-LABEL: func.func @pinned
// CHECK:       module @outlined
// CHECK:         func.func @pinned_host0(%[[S:.*]]: tensor<4xi32>, %{{.*}}: tensor<4xi32>) -> tensor<4xi32>
// CHECK-NOT:       bufferization.materialize_in_destination
// CHECK:           return %[[S]]
func.func @pinned(%x: tensor<4xi32>, %d: tensor<4xi32>) -> tensor<4xi32> {
  %y = cinm.compute -> tensor<4xi32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
    %p = bufferization.materialize_in_destination %x in %d : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
    cinm.yield %p : tensor<4xi32>
  }
  return %y : tensor<4xi32>
}
