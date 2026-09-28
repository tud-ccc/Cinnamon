// RUN: cinm-opt %s --split-input-file --cinm-isolate-compute-blocks --upmem-infer-accelerator="simulator=fast max-evals=4 n-init=2 graph-allocation=true latency-objective=true allow-host-placement=true dump-dir=%t" | FileCheck %s
// RUN: FileCheck %s --input-file=%t/infer_slow_host/allocation.json --check-prefix=SLOW
// RUN: FileCheck %s --input-file=%t/infer_fast_host/allocation.json --check-prefix=FAST
// RUN: FileCheck %s --input-file=%t/infer_slow_host/class_0/reference.mlir --check-prefix=REF
// RUN: FileCheck %s --input-file=%t/infer_unpriced/allocation.json --check-prefix=UNPRICED

// Where a block runs is the allocation's decision, not a screen's: every
// class is offered a point that leaves it on the host, priced by the host's
// roofline, and the latency solve spends the device where it buys the most
// makespan. The two halves of that decision, on one kernel and two hosts.

// A host that streams 1 GB/s and computes 1 Gop/s takes 8.4 ms over this
// matvec's 4 MB of weights; 1024 DPUs take a fraction of that, so the
// allocation puts it there.
//
// The allocation report shows the decision: the host is point 0 of the
// class, and the group the solve gave it holds devices.
//
// SLOW: "groups": [
// SLOW: "on_host": false,
// SLOW: "points": [
// SLOW: "priced_by": "host_roofline",
// SLOW: "where": "host"
// SLOW: "priced_by": "search",
// SLOW: "reference": {
// SLOW-NEXT: "function": "infer_slow_host_class0",
// SLOW-NEXT: "path": "class_0/reference.mlir"
//
// Each class's reference module is dumped beside the report: its one
// compute block in the form the space is read off, the weights still
// static, and the host it is priced against, so that a configuration from
// the report can be compiled from it alone.
//
// REF: func.func @infer_slow_host_class0(%{{.*}}: tensor<2048x2048xi8> {cinm.static}, %{{.*}}: tensor<2048xi8>)
// REF-SAME: cinm.available_platforms = [#cinm.host_platform<ops_per_second = 1.000000e+09
// REF: cinm.compute_block
// REF: linalg.generic
//
// CHECK-LABEL: func.func @slow_host
// CHECK: upmem.alloc_dpus
// CHECK: cinm.compute_block on accelerator #upmem.array<
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 16>

func.func @slow_host(%A: tensor<2048x2048xi8> {cinm.static}, %x: tensor<2048xi8>) -> tensor<2048xi32>
    attributes {cinm.available_platforms = [
      #cinm.host_platform<ops_per_second = 1.0e9, dram_bytes_per_second = 1.0e9>,
      #upmem]} {
  %r = cinm.compute -> tensor<2048xi32> attributes {cinm.available_platforms = [#upmem]} {
    %e = tensor.empty() : tensor<2048xi32>
    %g = linalg.matvec ins(%A, %x : tensor<2048x2048xi8>, tensor<2048xi8>) outs(%e : tensor<2048xi32>) -> tensor<2048xi32>
    cinm.yield %g : tensor<2048xi32>
  }
  return %r : tensor<2048xi32>
}

// -----

// The same kernel against the machine this compiler is calibrated for: the
// host reads those 4 MB from DRAM faster than the array can be filled with
// them, so no size of device pays and the block stays where it is. It keeps
// its graph_alloc record, which says so.
//
// FAST: "groups": [
// FAST: "on_host": true,
// FAST: "point_resource": 0,
// FAST: "resource": 0,
// FAST: "points": [
// FAST: "in_profile": true,
// FAST-NEXT: "priced_by": "host_roofline",
//
// CHECK-LABEL: func.func @fast_host
// CHECK-NOT: upmem.alloc_dpus
// CHECK: cinm.compute_block (
// CHECK-SAME: placement = "host"
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 16>

func.func @fast_host(%A: tensor<2048x2048xi8> {cinm.static}, %x: tensor<2048xi8>) -> tensor<2048xi32>
    attributes {cinm.available_platforms = [#cinm.host_platform, #upmem]} {
  %r = cinm.compute -> tensor<2048xi32> attributes {cinm.available_platforms = [#upmem]} {
    %e = tensor.empty() : tensor<2048xi32>
    %g = linalg.matvec ins(%A, %x : tensor<2048x2048xi8>, tensor<2048xi8>) outs(%e : tensor<2048xi32>) -> tensor<2048xi32>
    cinm.yield %g : tensor<2048xi32>
  }
  return %r : tensor<2048xi32>
}

// -----

// A block whose host cost cannot be read -- a loop in its body runs a number
// of times only known at run time, so its footprint is unknown -- has no host
// point to offer the allocation. Its device points would then be chosen
// against nothing, whatever they cost; it stays on the host instead, and the
// report says why.
//
// UNPRICED: "fate": "host_unpriced",
//
// CHECK-LABEL: func.func @unpriced
// CHECK-NOT: upmem.alloc_dpus
// CHECK: cinm.compute_block (
// CHECK-NOT: upmem.alloc_dpus
// CHECK: return
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 16>
#map = affine_map<(d0) -> (d0)>

func.func @unpriced(%x: tensor<65536xi32>, %steps: index) -> tensor<65536xi32>
    attributes {cinm.available_platforms = [#cinm.host_platform, #upmem]} {
  %r = cinm.compute -> tensor<65536xi32> attributes {cinm.available_platforms = [#upmem]} {
    %e = tensor.empty() : tensor<65536xi32>
    %g = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
        ins(%x : tensor<65536xi32>) outs(%e : tensor<65536xi32>) {
    ^bb0(%n: i32, %o: i32):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : i32
      %y = scf.for %k = %c0 to %steps step %c1 iter_args(%y0 = %n) -> (i32) {
        %h = arith.divsi %y0, %c2 : i32
        scf.yield %h : i32
      }
      linalg.yield %y : i32
    } -> tensor<65536xi32>
    cinm.yield %g : tensor<65536xi32>
  }
  return %r : tensor<65536xi32>
}
