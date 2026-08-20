// RUN: cinm-opt --upmem-annotate-costs=simulator=cycle-accurate %s | FileCheck %s

// A tiled inner loop whose bounds both move with the enclosing induction
// variable -- `scf.for %j = %t to %t + 2` -- has a trip count of 2 even though
// neither endpoint is a constant. Reading the endpoints alone leaves the count
// at 0, and the simulator then charges 2^32 repeats for it, by subtracting the
// one repeat it has already run from a count of none.
//
// The two kernels below are the same program written with the two bound forms,
// down to the constants they declare, so they have to cost the same. Comparing
// them rather than pinning a number keeps the test honest across recalibration
// of the latency tables.

// CHECK: upmem.wait_for
// CHECK-SAME: upmem.sim_cost = [[COST:[0-9][0-9.e+-]*]] : f64
// CHECK: upmem.wait_for
// CHECK-SAME: upmem.sim_cost = [[COST]] : f64

#upmem = #upmem.platform<type = v1A, dpus = 64, tasklets = 1>
module {
  // The reference: the same loop nest with a literal trip count.
  func.func @constant_bounds() attributes {cinm.available_platforms = [#upmem]} {
    cinm.compute_block on accelerator #upmem.array<64x1, <type = v1A, dpus = 64, tasklets = 1>> () attributes {cinm.available_platforms = [#upmem]} {
      %0 = upmem.alloc_dpus : !upmem.hierarchy<64x1>
      upmem.load_program @dpu_kernels_0::@const_program on %0 : !upmem.hierarchy<64x1>
      upmem.wait_for %0 : !upmem.hierarchy<64x1>
      upmem.free_dpus %0 : !upmem.hierarchy<64x1>
      cinm.yield
    }
    return
  }

  func.func @affine_bounds() attributes {cinm.available_platforms = [#upmem]} {
    cinm.compute_block on accelerator #upmem.array<64x1, <type = v1A, dpus = 64, tasklets = 1>> () attributes {cinm.available_platforms = [#upmem]} {
      %0 = upmem.alloc_dpus : !upmem.hierarchy<64x1>
      upmem.load_program @dpu_kernels_0::@affine_program on %0 : !upmem.hierarchy<64x1>
      upmem.wait_for %0 : !upmem.hierarchy<64x1>
      upmem.free_dpus %0 : !upmem.hierarchy<64x1>
      cinm.yield
    }
    return
  }

  module @dpu_kernels_0 {
    upmem.dpu_program @const_program() tasklets(1) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %alloca = memref.alloca() : memref<1x16xi32, #upmem.wram>
      %mram_buf = upmem.static_alloc @buf(mram) noinit : memref<128x16xi32, #upmem.mram>
      scf.for %arg0 = %c0 to %c64 step %c1 {
        %0 = arith.muli %arg0, %c2 : index
        %1 = arith.addi %0, %c2 : index
        scf.for %arg1 = %c0 to %c2 step %c1 {
          %subview = memref.subview %mram_buf[%arg1, 0] [1, 16] [1, 1] : memref<128x16xi32, #upmem.mram> to memref<1x16xi32, strided<[16, 1], offset: ?>, #upmem.mram>
          upmem.local_transfer %subview into %alloca : memref<1x16xi32, strided<[16, 1], offset: ?>, #upmem.mram> to memref<1x16xi32, #upmem.wram>
          %2 = memref.load %alloca[%c0, %c0] : memref<1x16xi32, #upmem.wram>
          %3 = arith.addi %2, %2 : i32
          memref.store %3, %alloca[%c0, %c0] : memref<1x16xi32, #upmem.wram>
        }
      }
      upmem.return
    }

    upmem.dpu_program @affine_program() tasklets(1) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %alloca = memref.alloca() : memref<1x16xi32, #upmem.wram>
      %mram_buf = upmem.static_alloc @buf(mram) noinit : memref<128x16xi32, #upmem.mram>
      scf.for %arg0 = %c0 to %c64 step %c1 {
        %0 = arith.muli %arg0, %c2 : index
        %1 = arith.addi %0, %c2 : index
        scf.for %arg1 = %0 to %1 step %c1 {
          %subview = memref.subview %mram_buf[%arg1, 0] [1, 16] [1, 1] : memref<128x16xi32, #upmem.mram> to memref<1x16xi32, strided<[16, 1], offset: ?>, #upmem.mram>
          upmem.local_transfer %subview into %alloca : memref<1x16xi32, strided<[16, 1], offset: ?>, #upmem.mram> to memref<1x16xi32, #upmem.wram>
          %2 = memref.load %alloca[%c0, %c0] : memref<1x16xi32, #upmem.wram>
          %3 = arith.addi %2, %2 : i32
          memref.store %3, %alloca[%c0, %c0] : memref<1x16xi32, #upmem.wram>
        }
      }
      upmem.return
    }
  }
}
