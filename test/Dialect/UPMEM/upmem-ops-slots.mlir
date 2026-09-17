// RUN: cinm-opt %s | cinm-opt | FileCheck %s
// RUN: cinm-opt %s --mlir-print-op-generic | cinm-opt | FileCheck %s

// A slotted resident buffer and the transfers that address one of its
// slots round-trip through the printer and parser; a broadcast into a slot
// is checked against the slot's shape, not the whole buffer's.

#map = affine_map<(d) -> (d, 0)>

// CHECK-LABEL: func.func @slotted
func.func @slotted(%w: memref<8x1024xi8>, %x: memref<64xi8>, %slot: index) {
  %set = upmem.alloc_dpus : !upmem.hierarchy<8x1>
  upmem.load_program @dpu_kernels::@program on %set : !upmem.hierarchy<8x1>
  // CHECK: upmem.scatter_on_array %{{.*}}[1024 elts, #{{.*}}] onto @weights slot %{{.*}} of %{{.*}} : memref<8x1024xi8> onto !upmem.hierarchy<8x1>
  upmem.scatter_on_array %w[1024 elts, #map] onto @weights slot %slot of %set : memref<8x1024xi8> onto !upmem.hierarchy<8x1>
  // CHECK: upmem.broadcast %{{.*}} onto @bias slot %{{.*}} of %{{.*}} : memref<64xi8> onto !upmem.hierarchy<8x1>
  upmem.broadcast %x onto @bias slot %slot of %set : memref<64xi8> onto !upmem.hierarchy<8x1>
  upmem.free_dpus %set : !upmem.hierarchy<8x1>
  return
}

module @dpu_kernels {
  upmem.dpu_program @program() tasklets(1) {
    // CHECK: upmem.static_alloc @weights(mram) noinit slots 4 : memref<4x1024xi8, "mram">
    %w = upmem.static_alloc @weights(mram) noinit slots 4 : memref<4x1024xi8, "mram">
    // CHECK: upmem.static_alloc @bias(mram) noinit slots 4 : memref<4x64xi8, "mram">
    %b = upmem.static_alloc @bias(mram) noinit slots 4 : memref<4x64xi8, "mram">
    upmem.return
  }
}
