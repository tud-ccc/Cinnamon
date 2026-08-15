// RUN: cinm-opt "--upmem-annotate-costs=simulator=op-count costs-csv=%t.csv" %s -o /dev/null
// RUN: FileCheck %s < %t.csv

// A transfer whose data is the same on every inference (upmem.timing_tag
// says static:) and which runs once per invocation is paid at load time by a
// serving deployment, so it is reported but not charged: it lands in an
// `excluded` row of its own, and the block total counts the other two
// scatters only -- the dynamic one, and the static one under a loop, which
// moves a different tile every trip and so cannot be hoisted to load time.

// CHECK: block_id,location,category,label,cost_ms,excluded,block_total_ms
// CHECK-NEXT: transfer,"blocks",{{[0-9.e+-]+}},1,[[TOTAL:[0-9.e+-]+]]
// CHECK-NEXT: transfer,"blocks",[[TOTAL]],0,[[TOTAL]]

#upmem = #upmem.platform<type = v1A, dpus = 512, tasklets = 8>
module {
  func.func @f(%arg0: memref<128x32xi32>, %arg1: memref<128x32xi32>, %arg2: memref<128x32xi32>) attributes {cinm.available_platforms = [#upmem]} {
    %r = cinm.compute_block on accelerator #upmem.array<128x4, <type = v1A, dpus = 512, tasklets = 8>> (%a = %arg0 : memref<128x32xi32>, %b = %arg1 : memref<128x32xi32>, %c = %arg2 : memref<128x32xi32>) -> memref<128x32xi32> attributes {cinm.available_platforms = [#upmem]} {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %1 = upmem.alloc_dpus : !upmem.hierarchy<128x4>
      upmem.load_program @dpu_kernels::@program on %1 : !upmem.hierarchy<128x4>
      upmem.scatter_blocks %a[8 elts, affine_map<(d0, d1) -> (d0, d1 * 8)>, 4 blocks] onto @buf of %1
          {upmem.timing_tag = "static:0"} : memref<128x32xi32> onto !upmem.hierarchy<128x4>
      upmem.scatter_blocks %b[8 elts, affine_map<(d0, d1) -> (d0, d1 * 8)>, 4 blocks] onto @buf of %1
          {upmem.timing_tag = "dyn:1"} : memref<128x32xi32> onto !upmem.hierarchy<128x4>
      scf.for %i = %c0 to %c4 step %c1 {
        upmem.scatter_blocks %c[8 elts, affine_map<(d0, d1) -> (d0, d1 * 8)>, 4 blocks] onto @buf of %1
            {upmem.timing_tag = "static:2"} : memref<128x32xi32> onto !upmem.hierarchy<128x4>
      }
      upmem.free_dpus %1 : !upmem.hierarchy<128x4>
      cinm.yield %a : memref<128x32xi32>
    }
    return
  }
  module @dpu_kernels {
    upmem.dpu_program @program() tasklets(4) {
      %buf = upmem.static_alloc @buf(mram) : memref<4x8xi32, "mram">
      %wram = upmem.static_alloc(wram) : memref<4x8xi32, "wram">
      upmem.local_transfer %buf into %wram : memref<4x8xi32, "mram"> to memref<4x8xi32, "wram">
      upmem.return
    }
  }
}
