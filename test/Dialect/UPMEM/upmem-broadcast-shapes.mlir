// RUN: cinm-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// A broadcast copies the host buffer verbatim into each DPU's buffer, so what
// has to agree is the sequence of elements, not the shape. The two sides group
// that sequence differently: an operand the workgroup shares along a
// distributed dimension keeps that dimension separate on the host, so a
// 16-tasklet buffer whose iteration dimension is split two ways arrives shaped
// (2, 8, ...) against a device buffer of (16, ...).

// CHECK-LABEL: @grouped_dims_are_the_same_buffer
func.func @grouped_dims_are_the_same_buffer(%host: memref<2x8x32x32xi32>) {
  %h = upmem.alloc_dpus : !upmem.hierarchy<2048x16>
  upmem.load_program @dpu_kernels::@program on %h : !upmem.hierarchy<2048x16>
  // CHECK: upmem.broadcast
  upmem.broadcast %host onto @buf of %h : memref<2x8x32x32xi32> onto !upmem.hierarchy<2048x16>
  return
}

module @dpu_kernels {
  upmem.dpu_program @program() tasklets(16) {
    %b = upmem.static_alloc @buf(mram) noinit : memref<16x32x1x32xi32, #upmem.mram>
    upmem.return
  }
}

// -----

// Unit dimensions are ignored on either side.

// CHECK-LABEL: @unit_dims_are_ignored
func.func @unit_dims_are_ignored(%host: memref<1x16x1x1024xi32>) {
  %h = upmem.alloc_dpus : !upmem.hierarchy<2048x16>
  upmem.load_program @dpu_kernels::@program on %h : !upmem.hierarchy<2048x16>
  // CHECK: upmem.broadcast
  upmem.broadcast %host onto @buf of %h : memref<1x16x1x1024xi32> onto !upmem.hierarchy<2048x16>
  return
}

module @dpu_kernels {
  upmem.dpu_program @program() tasklets(16) {
    %b = upmem.static_alloc @buf(mram) noinit : memref<16x32x1x32xi32, #upmem.mram>
    upmem.return
  }
}

// -----

// Grouping is not reordering. These hold the same elements in a different
// order, and a verbatim copy would move the right bytes to the wrong places.

func.func @transposed_shapes_are_rejected(%host: memref<32x16xi32>) {
  %h = upmem.alloc_dpus : !upmem.hierarchy<2048x16>
  upmem.load_program @dpu_kernels::@program on %h : !upmem.hierarchy<2048x16>
  // expected-error @below {{host buffer shape 'memref<32x16xi32>' is not compatible with target buffer 'memref<16x32xi32, #upmem.mram>'}}
  upmem.broadcast %host onto @buf of %h : memref<32x16xi32> onto !upmem.hierarchy<2048x16>
  return
}

module @dpu_kernels {
  upmem.dpu_program @program() tasklets(16) {
    %b = upmem.static_alloc @buf(mram) noinit : memref<16x32xi32, #upmem.mram>
    upmem.return
  }
}

// -----

// Nor is it a licence to move a different number of elements.

func.func @a_short_buffer_is_rejected(%host: memref<16x16xi32>) {
  %h = upmem.alloc_dpus : !upmem.hierarchy<2048x16>
  upmem.load_program @dpu_kernels::@program on %h : !upmem.hierarchy<2048x16>
  // expected-error @below {{host buffer shape 'memref<16x16xi32>' is not compatible with target buffer 'memref<16x32xi32, #upmem.mram>'}}
  upmem.broadcast %host onto @buf of %h : memref<16x16xi32> onto !upmem.hierarchy<2048x16>
  return
}

module @dpu_kernels {
  upmem.dpu_program @program() tasklets(16) {
    %b = upmem.static_alloc @buf(mram) noinit : memref<16x32xi32, #upmem.mram>
    upmem.return
  }
}
