// RUN: cinm-opt %s --cinm-isolate-compute-blocks --upmem-infer-accelerator="simulator=op-count max-evals=2 n-init=2 graph-allocation=true allocation-granularity=4" | FileCheck %s

// The graph-level allocation with hoisted device sets: the two blocks have
// different shapes, so program identity forbids merging them and each gets
// its own pinned group. Each group's set is allocated ONCE at the top of
// the container function and forwarded into the member block as an
// operand; the block loads its program on the forwarded argument, does not
// allocate a set of its own, and does not free what it does not own. The
// frees sit at the function exit, where the residency lifetime ends.
//
// The checks are structural (which ops appear where), not numeric: the
// exact split of the 16 DPUs between the two groups is the cost model's
// business and may change with it.

#upmem = #upmem.platform<type = v1A, dpus = 16, tasklets = 16>

// CHECK-LABEL: func.func @two_classes
// CHECK-DAG: %[[WG0:[0-9]+]] = upmem.alloc_dpus : !upmem.hierarchy<
// CHECK-DAG: %[[WG1:[0-9]+]] = upmem.alloc_dpus : !upmem.hierarchy<
// CHECK: cinm.compute_block on accelerator
// CHECK-SAME: !upmem.hierarchy<
// CHECK-NOT: upmem.alloc_dpus
//
// The commit marks casts of operands the body only reads (per the transfer
// ops' declared effects) as read_only, so the enclosing function's later
// bufferization does not pay a defensive alloc+copy for them. Both gemv
// inputs are scattered, never gathered into, so both casts qualify.
// CHECK: bufferization.to_buffer %arg{{[0-9]+}} read_only
// CHECK: bufferization.to_buffer %arg{{[0-9]+}} read_only
// CHECK: upmem.load_program @{{.*}} on %arg{{[0-9]+}} : !upmem.hierarchy<
// CHECK-NOT: upmem.alloc_dpus
// CHECK-NOT: upmem.free_dpus
// CHECK: cinm.compute_block on accelerator
// CHECK-SAME: !upmem.hierarchy<
// CHECK-NOT: upmem.alloc_dpus
// CHECK: upmem.load_program @{{.*}} on %arg{{[0-9]+}} : !upmem.hierarchy<
// CHECK-NOT: upmem.alloc_dpus
// CHECK: upmem.free_dpus
// CHECK: upmem.free_dpus
// CHECK: return
func.func @two_classes(%A: tensor<256x256xi32>, %B: tensor<128x256xi32>, %x: tensor<256xi32>) -> tensor<128xi32> {
  %r = cinm.compute -> tensor<256xi32> attributes {cinm.available_platforms = [#upmem]} {
    %g = cinm.op.gemv %A, %x : tensor<256x256xi32>, tensor<256xi32> -> tensor<256xi32>
    cinm.yield %g : tensor<256xi32>
  }
  %r2 = cinm.compute -> tensor<128xi32> attributes {cinm.available_platforms = [#upmem]} {
    %g = cinm.op.gemv %B, %r : tensor<128x256xi32>, tensor<256xi32> -> tensor<128xi32>
    cinm.yield %g : tensor<128xi32>
  }
  return %r2 : tensor<128xi32>
}
