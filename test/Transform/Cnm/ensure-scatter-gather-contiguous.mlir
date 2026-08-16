// RUN: cinm-opt %s --cnm-ensure-scatter-gather-contiguous | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
// A gather has to partition its destination, so unlike a scatter it cannot
// ignore the tasklet dimension.
#gmap = affine_map<(d0, d1, d2) -> (d0 * 16 + d1, d2)>
#upmem_2_4_16 = #upmem.array<8x16, <type = v1A, dpus = 4096, tasklets = 1>>

// A scatter whose input is a strided subview (each of the 8 rows of 256
// elements sits inside a wider 512-element row) is not safe to transfer with
// a single flat memcpy. The pass must insert a packing alloc + repack and
// rewire the scatter to use it.
//
// The repack is a cnm.compact_buffer rather than a memref.copy so the backend
// can route it somewhere it is timed: a memref.copy between two contiguous
// memrefs lowers to llvm.intr.memcpy and would never show up in a
// measurement. Its map is the identity -- the shape is unchanged and only the
// layout is -- and being emitted first it takes #map, moving the scatter's
// own map to #map1.

// CHECK-LABEL: func.func @scatter_noncontiguous
// CHECK:       %[[BIG:.*]] = memref.alloc() : memref<8x512xi32>
// CHECK:       %[[VIEW:.*]] = memref.subview %[[BIG]]
// CHECK:       %[[PACK:.*]] = memref.get_global @{{.*}} : memref<8x256xi32>
// CHECK-NEXT:  cnm.compact_buffer %[[VIEW]] into %[[PACK]][#map] : memref<8x256xi32, strided<[512, 1]>> into memref<8x256xi32>
// CHECK-NEXT:  cnm.scatter %[[PACK]] into %{{.*}}[#map1] of %{{.*}} : memref<8x256xi32> into
func.func @scatter_noncontiguous() {
  %wg = cnm.workgroup : !cnm.workgroup<#upmem_2_4_16>
  %buf = cnm.declare_buffer() for %wg : !cnm.buffer<256xi32 on #upmem_2_4_16>
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
// CHECK-NEXT:  cnm.scatter %[[BUF]] into %{{.*}}[#map1] of %{{.*}}
// CHECK-NOT:   cnm.compact_buffer
func.func @scatter_already_contiguous() {
  %wg = cnm.workgroup : !cnm.workgroup<#upmem_2_4_16>
  %buf = cnm.declare_buffer() for %wg : !cnm.buffer<256xi32 on #upmem_2_4_16>
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
// CHECK:       %[[BIG:.*]] = memref.alloc() : memref<128x512xi32>
// CHECK:       %[[VIEW:.*]] = memref.subview %[[BIG]]
// The repack buffer is a module-level global, reused by every call rather
// than allocated per call -- and so not freed after the copy either.
// CHECK:       %[[PACK:.*]] = memref.get_global @{{.*}} : memref<128x256xi32>
// CHECK-NEXT:  cnm.gather %{{.*}}[#map2] of %{{.*}} into %[[PACK]] : {{.*}} into memref<128x256xi32>
// CHECK-NEXT:  memref.copy %[[PACK]], %[[VIEW]]
// CHECK-NOT:   memref.dealloc
func.func @gather_noncontiguous() {
  %wg = cnm.workgroup : !cnm.workgroup<#upmem_2_4_16>
  %buf = cnm.declare_buffer() for %wg : !cnm.buffer<256xi32 on #upmem_2_4_16>
  %big = memref.alloc() : memref<128x512xi32>
  %view = memref.subview %big[0, 0] [128, 256] [1, 1]
      : memref<128x512xi32> to memref<128x256xi32, strided<[512, 1], offset: 0>>
  cnm.gather %buf[#gmap] of %wg into %view
      : !cnm.buffer<256xi32 on #upmem_2_4_16> into memref<128x256xi32, strided<[512, 1], offset: 0>>
  cnm.free_workgroup %wg : !cnm.workgroup<#upmem_2_4_16>
  return
}

// A gather whose destination is already fully packed must be left untouched.

// CHECK-LABEL: func.func @gather_already_contiguous
// CHECK:       %[[OUT:.*]] = memref.alloc() : memref<128x256xi32>
// CHECK-NEXT:  cnm.gather %{{.*}}[#map2] of %{{.*}} into %[[OUT]]
// CHECK-NOT:   memref.copy
func.func @gather_already_contiguous() {
  %wg = cnm.workgroup : !cnm.workgroup<#upmem_2_4_16>
  %buf = cnm.declare_buffer() for %wg : !cnm.buffer<256xi32 on #upmem_2_4_16>
  %out = memref.alloc() : memref<128x256xi32>
  cnm.gather %buf[#gmap] of %wg into %out
      : !cnm.buffer<256xi32 on #upmem_2_4_16> into memref<128x256xi32>
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
  %buf = cnm.declare_buffer() for %wg : !cnm.buffer<256xi32 on #upmem_2_4_16>
  cnm.scatter %arg0 into %buf[#map] of %wg
      : tensor<8x256xi32> into !cnm.buffer<256xi32 on #upmem_2_4_16>
  cnm.free_workgroup %wg : !cnm.workgroup<#upmem_2_4_16>
  return
}
