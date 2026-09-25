// RUN: cinm-opt %s --cinm-isolate-compute-blocks --upmem-infer-accelerator="simulator=op-count max-evals=2 n-init=2 graph-allocation=true menu-screen-csv=%t gate-dry-run=true" -o /dev/null
// RUN: FileCheck %s --input-file=%t/infer_gemv_menu_screen.csv --check-prefix=CSV
// RUN: cinm-opt %s --cinm-isolate-compute-blocks --upmem-infer-accelerator="simulator=op-count max-evals=2 n-init=2 graph-allocation=true screen-menu=true" 2>&1 | FileCheck %s --check-prefix=SCREEN

// The menu screen prices this block's roofline at every DPU count the block
// admits and keeps only the counts that beat the host. This gemv streams a
// per-inference matrix and holds nothing resident, so the host reads it from
// DRAM once while the device would have to send all of it over the wire --
// no count can win, and the block never enters a search at all.
//
// The dry run reports that without deciding anything: one row per candidate
// count, both rooflines, and the verdict.
//
// CSV: graph,class,blocks,loc,work_ops,static_bytes,dynamic_bytes,host_ms,resource,device_ms,kept,candidates,kept_of_candidates,profiled
// CSV: "infer_gemv",0,1,
// CSV-NOT: ,1,{{[0-9]+}},{{[0-9]+}},1{{$}}

// SCREEN: no resource value beats the host on this block
// SCREEN-NOT: cinm.compute_block on accelerator #upmem.array

#upmem = #upmem.platform<type = v1A, dpus = 16, tasklets = 16>

func.func @gemv(%A: tensor<256x256xi32>, %x: tensor<256xi32>) -> tensor<256xi32> {
  %r = cinm.compute -> tensor<256xi32> attributes {cinm.available_platforms = [#upmem]} {
    %g = cinm.op.gemv %A, %x : tensor<256x256xi32>, tensor<256xi32> -> tensor<256xi32>
    cinm.yield %g : tensor<256xi32>
  }
  return %r : tensor<256xi32>
}
