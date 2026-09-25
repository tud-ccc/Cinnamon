// RUN: cinm-opt %s --split-input-file --cinm-isolate-compute-blocks --upmem-infer-accelerator="simulator=fast max-evals=4 n-init=2 graph-allocation=true latency-objective=true allow-host-placement=true" | FileCheck %s

// Where a block runs is the allocation's decision, not a screen's: every
// class is offered a point that leaves it on the host, priced by the host's
// roofline, and the latency solve spends the device where it buys the most
// makespan. The two halves of that decision, on one kernel and two hosts.

// A host that streams 1 GB/s and computes 1 Gop/s takes 8.4 ms over this
// matvec's 4 MB of weights; 1024 DPUs take a fraction of that, so the
// allocation puts it there.
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
