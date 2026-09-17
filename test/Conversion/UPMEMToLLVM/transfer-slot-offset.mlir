// RUN: cinm-opt %s --convert-upmem-to-llvm | FileCheck %s

// A transfer into a slot of a slotted buffer hands the runtime the slot's
// byte offset in the symbol -- slot times the slot size, here 1024 i8 --
// after the buffer id; a transfer without a slot passes zero.

#map = affine_map<(d) -> (d, 0)>

// CHECK-LABEL: func.func @scatter_into_slot
//       CHECK:   %[[SLOT:.*]] = builtin.unrealized_conversion_cast %{{.*}} : index to i64
//       CHECK:   %[[ID:.*]] = llvm.mlir.addressof @buffer_name0
//       CHECK:   %[[SIZE:.*]] = llvm.mlir.constant(1024 : i64)
//       CHECK:   %[[OFF:.*]] = llvm.mul %[[SLOT]], %[[SIZE]]
//       CHECK:   llvm.call @upmemrt_dpu_scatter(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[ID]], %[[OFF]], %{{.*}}, %{{.*}})
//       CHECK:   %[[BIAS:.*]] = llvm.mlir.addressof @buffer_name1
//       CHECK:   %[[ZERO:.*]] = llvm.mlir.constant(0 : i64)
//       CHECK:   llvm.call @upmemrt_dpu_broadcast(%{{.*}}, %{{.*}}, %{{.*}}, %[[BIAS]], %[[ZERO]], %{{.*}})
func.func @scatter_into_slot(%w: memref<8x1024xi8>, %x: memref<64xi8>, %slot: index) {
  %set = upmem.alloc_dpus : !upmem.hierarchy<8x1>
  upmem.load_program @dpu_kernels::@program on %set : !upmem.hierarchy<8x1>
  upmem.scatter_on_array %w[1024 elts, #map] onto @weights slot %slot of %set : memref<8x1024xi8> onto !upmem.hierarchy<8x1>
  upmem.broadcast %x onto @bias of %set : memref<64xi8> onto !upmem.hierarchy<8x1>
  upmem.free_dpus %set : !upmem.hierarchy<8x1>
  return
}

module @dpu_kernels {
  upmem.dpu_program @program() tasklets(1) {
    %w = upmem.static_alloc @weights(mram) noinit slots 4 : memref<4x1024xi8, "mram">
    %b = upmem.static_alloc @bias(mram) noinit : memref<64xi8, "mram">
    upmem.return
  }
}
