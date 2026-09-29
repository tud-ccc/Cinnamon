// RUN: cinm-opt %s --cinm-isolate-compute-blocks --upmem-infer-accelerator="simulator=fast fixed-tasklets=4 graph-allocation=1 allocation-in=%S/Inputs/allocation-in" | FileCheck %s
// RUN: not cinm-opt %s --cinm-isolate-compute-blocks --upmem-infer-accelerator="simulator=fast fixed-tasklets=4 graph-allocation=1 allocation-in=%S/Inputs/allocation-in-unlisted" 2>&1 | FileCheck %s --check-prefix=UNLISTED
// RUN: not cinm-opt %s --cinm-isolate-compute-blocks --upmem-infer-accelerator="simulator=fast fixed-tasklets=4 graph-allocation=1 allocation-in=%S/Inputs/allocation-in-mismatch" 2>&1 | FileCheck %s --check-prefix=MISMATCH
// RUN: not cinm-opt %s --cinm-isolate-compute-blocks --upmem-infer-accelerator="simulator=fast fixed-tasklets=4 allocation-in=%S/Inputs/allocation-in" 2>&1 | FileCheck %s --check-prefix=NOGRAPH

// allocation-in commits the allocation a file gives instead of profiling and
// solving: the first gemv runs on its group of 16 DPUs with the file's
// configuration, the second stays on the host, whatever a solve would have
// chosen.

// CHECK-LABEL: func.func @chained
// CHECK: cinm.compute_block on accelerator #upmem.array<16x4
// CHECK-SAME: cinm.graph_alloc = {class = 0 : i64, graph = "infer_chained", group = 0 : i64
// CHECK: cinm.graph_alloc = {class = 1 : i64, graph = "infer_chained", member = 0 : i64, placement = "host"}
// CHECK-NOT: on accelerator

// NOGRAPH: error: allocation-in is committed by graph allocation only
// UNLISTED: error: allocation-in: class 1 is not listed
// MISMATCH: error: allocation-in: class 0: a group on 8 runs a config with dpus=16

#upmem = #upmem.platform<type = v1A, dpus = 16, tasklets = 16>

func.func @chained(%A: tensor<256x256xi32>, %x: tensor<256xi32>, %B: tensor<128x128xi32>) -> tensor<128xi32>
    attributes {cinm.available_platforms = [#upmem]} {
  %r = cinm.compute -> tensor<256xi32> {
    %g = cinm.op.gemv %A, %x : tensor<256x256xi32>, tensor<256xi32> -> tensor<256xi32>
    cinm.yield %g : tensor<256xi32>
  }
  %s = tensor.extract_slice %r[0] [128] [1] : tensor<256xi32> to tensor<128xi32>
  %r2 = cinm.compute -> tensor<128xi32> {
    %g = cinm.op.gemv %B, %s : tensor<128x128xi32>, tensor<128xi32> -> tensor<128xi32>
    cinm.yield %g : tensor<128xi32>
  }
  return %r2 : tensor<128xi32>
}
