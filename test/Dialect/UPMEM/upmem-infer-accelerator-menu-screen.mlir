// RUN: cinm-opt %s --cinm-isolate-compute-blocks --upmem-infer-accelerator="simulator=fast max-evals=2 n-init=2 graph-allocation=true allocation-report=%t gate-dry-run=true" -o /dev/null
// RUN: FileCheck %s --input-file=%t/infer_gemv.json --check-prefix=REPORT
// RUN: cinm-opt %s --cinm-isolate-compute-blocks --upmem-infer-accelerator="simulator=op-count max-evals=2 n-init=2 graph-allocation=true screen-menu=true" 2>&1 | FileCheck %s --check-prefix=SCREEN

// The menu screen prices this block's roofline at every DPU count the block
// admits and keeps only the counts that beat the host. This gemv streams a
// per-inference matrix and holds nothing resident, so the host reads it from
// DRAM once while the device would have to send all of it over the wire --
// no count can win, and the block never enters a search at all.
//
// The dry run reports that without deciding anything: the host as point 0,
// then every candidate count with its roofline and the verdict, and nothing
// allocated.
//
// REPORT: "allocation": null
// REPORT: "fate": "dry_run"
// REPORT: "addi(out0, muli(in0, in1))"
// REPORT: "kind": "cinm.op.gemv"
// REPORT: "priced_by": "host_roofline"
// REPORT: "where": "host"
// REPORT: "priced_by": "device_roofline"
// REPORT: "screen": "dropped"
// REPORT-NOT: "screen": "kept"
// REPORT-NOT: "selected": true
// REPORT: "graph": "infer_gemv"
// REPORT: "achieved_fraction": {{0\.64[0-9]*}}

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
