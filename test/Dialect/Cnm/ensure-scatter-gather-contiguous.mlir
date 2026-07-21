// RUN: cinm-opt %s --cnm-ensure-scatter-gather-contiguous | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0 * 4 + d1)>
#upmem_2_4_16 = #upmem.array<2x4x16, <type = v1A, dimensions = 32x128x1>>

// A scatter whose input is a strided subview (each of the 8 rows of 256
// elements sits inside a wider 512-element row) is not safe to transfer with
// a single flat memcpy. The pass must insert a packing alloc + copy and
// rewire the scatter to use it.

// CHECK-LABEL: func.func @scatter_noncontiguous
// CHECK:       %[[BIG:.*]] = memref.alloc() : memref<8x512xi32>
// CHECK:       %[[VIEW:.*]] = memref.subview %[[BIG]]
// CHECK:       %[[PACK:.*]] = memref.alloc() : memref<8x256xi32>
// CHECK-NEXT:  memref.copy %[[VIEW]], %[[PACK]]
// CHECK-NEXT:  cnm.scatter %[[PACK]] into %{{.*}}[#map] of %{{.*}} : memref<8x256xi32> into
func.func @scatter_noncontiguous() {
  %wg = cnm.workgroup : !cnm.workgroup<#upmem_2_4_16>
  %buf = cnm.alloc() for %wg : !cnm.buffer<256xi32 on #upmem_2_4_16>
  %big = memref.alloc() : memref<8x512xi32>
  %view = memref.subview %big[0, 0] [8, 256] [1, 1]
      : memref<8x512xi32> to memref<8x256xi32, strided<[512, 1], offset: 0>>
  cnm.scatter %view into %buf[#map] of %wg
      : memref<8x256xi32, strided<[512, 1], offset: 0>> into !cnm.buffer<256xi32 on #upmem_2_4_16>
  cnm.free_workgroup %wg : !cnm.workgroup<#upmem_2_4_16>
  return
}

// A scatter whose input is already fully packed must be left untouched.

// CHECK-LABEL: func.func @scatter_already_contiguous
// CHECK:       %[[BUF:.*]] = memref.alloc() : memref<8x256xi32>
// CHECK-NEXT:  cnm.scatter %[[BUF]] into %{{.*}}[#map] of %{{.*}}
// CHECK-NOT:   memref.copy
func.func @scatter_already_contiguous() {
  %wg = cnm.workgroup : !cnm.workgroup<#upmem_2_4_16>
  %buf = cnm.alloc() for %wg : !cnm.buffer<256xi32 on #upmem_2_4_16>
  %input = memref.alloc() : memref<8x256xi32>
  cnm.scatter %input into %buf[#map] of %wg
      : memref<8x256xi32> into !cnm.buffer<256xi32 on #upmem_2_4_16>
  cnm.free_workgroup %wg : !cnm.workgroup<#upmem_2_4_16>
  return
}

// A gather whose destination is a strided subview needs a packing buffer
// too, but the copy-back must happen *after* the gather (device -> packed
// buffer -> strided destination).

// CHECK-LABEL: func.func @gather_noncontiguous
// CHECK:       %[[BIG:.*]] = memref.alloc() : memref<8x512xi32>
// CHECK:       %[[VIEW:.*]] = memref.subview %[[BIG]]
// CHECK:       %[[PACK:.*]] = memref.alloc() : memref<8x256xi32>
// CHECK-NEXT:  cnm.gather %{{.*}}[#map] of %{{.*}} into %[[PACK]] : {{.*}} into memref<8x256xi32>
// CHECK-NEXT:  memref.copy %[[PACK]], %[[VIEW]]
// CHECK-NEXT:  memref.dealloc %[[PACK]]
func.func @gather_noncontiguous() {
  %wg = cnm.workgroup : !cnm.workgroup<#upmem_2_4_16>
  %buf = cnm.alloc() for %wg : !cnm.buffer<256xi32 on #upmem_2_4_16>
  %big = memref.alloc() : memref<8x512xi32>
  %view = memref.subview %big[0, 0] [8, 256] [1, 1]
      : memref<8x512xi32> to memref<8x256xi32, strided<[512, 1], offset: 0>>
  cnm.gather %buf[#map] of %wg into %view
      : !cnm.buffer<256xi32 on #upmem_2_4_16> into memref<8x256xi32, strided<[512, 1], offset: 0>>
  cnm.free_workgroup %wg : !cnm.workgroup<#upmem_2_4_16>
  return
}

// A gather whose destination is already fully packed must be left untouched.

// CHECK-LABEL: func.func @gather_already_contiguous
// CHECK:       %[[OUT:.*]] = memref.alloc() : memref<8x256xi32>
// CHECK-NEXT:  cnm.gather %{{.*}}[#map] of %{{.*}} into %[[OUT]]
// CHECK-NOT:   memref.copy
func.func @gather_already_contiguous() {
  %wg = cnm.workgroup : !cnm.workgroup<#upmem_2_4_16>
  %buf = cnm.alloc() for %wg : !cnm.buffer<256xi32 on #upmem_2_4_16>
  %out = memref.alloc() : memref<8x256xi32>
  cnm.gather %buf[#map] of %wg into %out
      : !cnm.buffer<256xi32 on #upmem_2_4_16> into memref<8x256xi32>
  cnm.free_workgroup %wg : !cnm.workgroup<#upmem_2_4_16>
  return
}

// Tensor (unbufferized) operands must be left alone: there is nothing to
// pack yet, that happens after bufferization.

// CHECK-LABEL: func.func @scatter_tensor_untouched
// CHECK-NOT: memref.alloc
// CHECK: cnm.scatter
func.func @scatter_tensor_untouched(%arg0: tensor<8x256xi32>) {
  %wg = cnm.workgroup : !cnm.workgroup<#upmem_2_4_16>
  %buf = cnm.alloc() for %wg : !cnm.buffer<256xi32 on #upmem_2_4_16>
  cnm.scatter %arg0 into %buf[#map] of %wg
      : tensor<8x256xi32> into !cnm.buffer<256xi32 on #upmem_2_4_16>
  cnm.free_workgroup %wg : !cnm.workgroup<#upmem_2_4_16>
  return
}
