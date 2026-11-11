// RUN: cinm-opt %s --upmem-time-compute-blocks | FileCheck %s
// RUN: cinm-opt %s --upmem-time-compute-blocks --cinm-unwrap-compute-blocks --convert-upmem-to-llvm | FileCheck %s --check-prefix=LLVM

// A device node of a solved graph -- a compute block on the UPMEM
// accelerator with a cinm.graph_alloc -- is bracketed with node timers named
// after its graph, class and member: the begin first in its body, the end
// right before its terminator. A block that was never allocated as part of a
// graph is not.

// CHECK-LABEL: func.func @timed
//       CHECK:   cinm.compute_block on accelerator #upmem.array
//  CHECK-NEXT:     upmem.node_begin "node:infer_g:3:1"
//  CHECK-NEXT:     memref.alloc
//  CHECK-NEXT:     upmem.node_end "node:infer_g:3:1"
//  CHECK-NEXT:     cinm.yield

// LLVM-DAG: llvm.mlir.global private constant @{{.*}}("node:infer_g:3:1\00")
// LLVM-DAG: llvm.func @upmemrt_node_begin(!llvm.ptr)
// LLVM-DAG: llvm.func @upmemrt_node_end(!llvm.ptr)
// LLVM-LABEL: func.func @timed
//       LLVM:   %[[BEGIN:.*]] = llvm.mlir.addressof
//  LLVM-NEXT:   llvm.call @upmemrt_node_begin(%[[BEGIN]])
//       LLVM:   memref.alloc
//       LLVM:   %[[END:.*]] = llvm.mlir.addressof
//  LLVM-NEXT:   llvm.call @upmemrt_node_end(%[[END]])
func.func @timed(%a: memref<16xi32>) -> memref<16xi32> {
  %0 = cinm.compute_block on accelerator #upmem.array<16x4, <type = v1A, dpus = 16, tasklets = 4>> (%x = %a : memref<16xi32>) -> memref<16xi32> attributes {cinm.graph_alloc = {class = 3 : i64, graph = "infer_g", member = 1 : i64}} {
    %b = memref.alloc() : memref<16xi32>
    cinm.yield %b : memref<16xi32>
  }
  return %0 : memref<16xi32>
}

// CHECK-LABEL: func.func @untimed
//   CHECK-NOT:   upmem.node_begin
//       CHECK:   return
// LLVM-LABEL: func.func @untimed
//   LLVM-NOT:   llvm.call @upmemrt_node
//       LLVM:   return
func.func @untimed(%a: memref<16xi32>) -> memref<16xi32> {
  %0 = cinm.compute_block on accelerator #upmem.array<16x4, <type = v1A, dpus = 16, tasklets = 4>> (%x = %a : memref<16xi32>) -> memref<16xi32> {
    %b = memref.alloc() : memref<16xi32>
    cinm.yield %b : memref<16xi32>
  }
  return %0 : memref<16xi32>
}
