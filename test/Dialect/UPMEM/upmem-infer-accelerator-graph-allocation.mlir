// RUN: cinm-opt %s --cinm-isolate-compute-blocks --upmem-infer-accelerator="simulator=op-count max-evals=4 n-init=4 fixed-tasklets=4 graph-allocation=1 allocation-granularity=4" | FileCheck %s --check-prefixes=CHECK,TPUT
// RUN: cinm-opt %s --cinm-isolate-compute-blocks --upmem-infer-accelerator="simulator=op-count max-evals=4 n-init=4 fixed-tasklets=4 graph-allocation=1 allocation-granularity=4 latency-objective=1" | FileCheck %s --check-prefixes=CHECK,LAT

// The graph-level two-level solve: profile each program-identity class over
// its menu (divisors of the iteration-space size, quantized to the
// allocation granularity), allocate the grid exactly, stamp each group's
// winning configuration onto its members. The platform has two ranks of
// 8 DPUs, so the grid budget is 16 and the menu here is {4, 8, 16}.
//
// What is pinned here is the *structure* of the outcome -- who shares a
// configuration, that the grid is not oversubscribed -- not which menu point
// wins: that is the solver's decision over measured costs.

#upmem = #upmem.platform<type = v1A, dpus = 16, tasklets = 16>

// Two different-shape gemvs: two classes, each a singleton. Both are pinned
// -- timesharing one of them would pay the 40 ms program reload against
// sub-ms kernels -- but the two objectives split the grid differently, and
// this is the case that shows why. Under throughput only the busiest set
// counts, so the large gemv takes 8 DPUs and the small one takes the
// *cheapest* size that stays under that bottleneck (4): widening it would
// make nothing faster. Under latency the two are chained, so the makespan is
// their sum and every DPU given to either one pays -- the grid goes 8/8.
// CHECK-LABEL: func.func @chained
func.func @chained(%A: tensor<256x256xi32>, %x: tensor<256xi32>, %B: tensor<128x128xi32>) -> tensor<128xi32>
    attributes {cinm.available_platforms = [#upmem]} {
  // CHECK: cinm.compute_block on accelerator #upmem.array<8x4
  %r = cinm.compute -> tensor<256xi32> {
    %g = cinm.op.gemv %A, %x : tensor<256x256xi32>, tensor<256xi32> -> tensor<256xi32>
    cinm.yield %g : tensor<256xi32>
  }
  %s = tensor.extract_slice %r[0] [128] [1] : tensor<256xi32> to tensor<128xi32>
  // TPUT: cinm.compute_block on accelerator #upmem.array<4x4
  // LAT: cinm.compute_block on accelerator #upmem.array<8x4
  %r2 = cinm.compute -> tensor<128xi32> {
    %g = cinm.op.gemv %B, %s : tensor<128x128xi32>, tensor<128xi32> -> tensor<128xi32>
    cinm.yield %g : tensor<128xi32>
  }
  return %r2 : tensor<128xi32>
}

// Two same-shape gemvs over different (static) weights: one class of two,
// and independent of each other. Program identity requires both members to
// commit the *same* configuration -- whether the solver co-locates them on
// one set or gives each its own equally-sized set, which is what the two
// objectives disagree about here.
//
// Being independent buys them nothing in time: the host blocks on every
// launch, so they run one after the other whichever sets they sit on (see
// makespanOf). Latency therefore merges them onto ONE set holding the whole
// grid -- two runs at full width beat two runs at half width -- while
// throughput, which only charges the busiest set, splits the grid and gives
// each its own. Both commit one shape, which is what the shared SHAPE
// capture below pins; the sizes differ between the two runs.
// CHECK-LABEL: func.func @qk
func.func @qk(%Wq: tensor<256x256xi32> {cinm.static}, %Wk: tensor<256x256xi32> {cinm.static}, %x: tensor<256xi32>)
    -> (tensor<256xi32>, tensor<256xi32>)
    attributes {cinm.available_platforms = [#upmem]} {
  // CHECK: cinm.compute_block on accelerator #upmem.array<[[SHAPE:[0-9x]+]],
  %q = cinm.compute -> tensor<256xi32> {
    %g = cinm.op.gemv %Wq, %x : tensor<256x256xi32>, tensor<256xi32> -> tensor<256xi32>
    cinm.yield %g : tensor<256xi32>
  }
  // CHECK: cinm.compute_block on accelerator #upmem.array<[[SHAPE]],
  %k = cinm.compute -> tensor<256xi32> {
    %g = cinm.op.gemv %Wk, %x : tensor<256x256xi32>, tensor<256xi32> -> tensor<256xi32>
    cinm.yield %g : tensor<256xi32>
  }
  return %q, %k : tensor<256xi32>, tensor<256xi32>
}
