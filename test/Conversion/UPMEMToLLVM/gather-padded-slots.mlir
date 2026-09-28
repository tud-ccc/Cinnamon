// RUN: cinm-opt %s --convert-upmem-to-llvm | FileCheck %s

// A gather over padded slots -- here each of three tasklets' slots holds two
// one-i32 blocks followed by three i32 of padding -- calls the padded entry
// point with the blocks per slot and the padding in bytes after the usual
// arguments. The DPU set is allocated for the transfer list the runtime
// builds: six blocks and one padding entry per slot, nine.

#map = affine_map<(d, b) -> (d, b)>

// CHECK-LABEL: func.func @gather_padded
//       CHECK:   %[[MAXB:.*]] = llvm.mlir.constant(9 : i64)
//       CHECK:   llvm.call @upmemrt_dpu_alloc_cached(%{{.*}}, %{{.*}}, %[[MAXB]])
//       CHECK:   %[[PERSLOT:.*]] = llvm.mlir.constant(2 : i64)
//       CHECK:   %[[PAD:.*]] = llvm.mlir.constant(12 : i64)
//       CHECK:   llvm.call @upmemrt_dpu_gather_blocks_padded({{.*}}, %[[PERSLOT]], %[[PAD]])
func.func @gather_padded(%out: memref<8x6xi32>) {
  %set = upmem.alloc_dpus : !upmem.hierarchy<8x3>
  upmem.load_program @dpu_kernels::@program on %set : !upmem.hierarchy<8x3>
  upmem.gather_blocks %out[1 elts, #map, 6 blocks] from @partials of %set {blocksPerSlot = 2 : i64, slotPadding = 3 : i64} : memref<8x6xi32> from !upmem.hierarchy<8x3>
  upmem.free_dpus %set : !upmem.hierarchy<8x3>
  return
}

module @dpu_kernels {
  upmem.dpu_program @program() tasklets(3) {
    %p = upmem.static_alloc @partials(mram) noinit : memref<3x5xi32, "mram">
    upmem.return
  }
}
