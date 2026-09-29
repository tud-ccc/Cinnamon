// RUN: cinm-opt %s --upmem-async-launches=noalias-arguments | FileCheck %s
// RUN: cinm-opt %s --upmem-async-launches=noalias-arguments --convert-upmem-to-llvm | FileCheck %s --check-prefix=LLVM

// Two gemv blocks on two DPU sets. Every launch and transfer is issued
// asynchronously; each set is synced right before the host code that reads
// what it gathered.

// In @par the blocks are independent, so the second block's issue moves
// above the first block's sync and host reduction: both sets work at once.
// CHECK-LABEL: func.func @par
// CHECK: upmem.wait_for %0 {upmem.async}
// CHECK: upmem.gather_from_array {{.*}} of %0 {upmem.async}
// CHECK: upmem.wait_for %[[S1:.*]] {upmem.async}
// CHECK: upmem.gather_from_array {{.*}} of %[[S1]] {upmem.async}
// CHECK: upmem.sync %0
// CHECK: affine.for
// CHECK: upmem.sync %[[S1]]
// CHECK: affine.for

// The issued ops lower to the runtime's asynchronous entry points.
// LLVM-DAG: llvm.call @upmemrt_dpu_launch_async
// LLVM-DAG: llvm.call @upmemrt_dpu_scatter_async
// LLVM-DAG: llvm.call @upmemrt_dpu_gather_async
// LLVM-DAG: llvm.call @upmemrt_dpu_sync

// In @seq the second block scatters what the first block's reduction wrote,
// so it stays after the first block's sync and reduction.
// CHECK-LABEL: func.func @seq
// CHECK: upmem.wait_for %0 {upmem.async}
// CHECK: upmem.sync %0
// CHECK: affine.for
// CHECK: upmem.scatter_on_array %{{.*}} onto @buf_1 of %[[S1:.*]] {upmem.async}
// CHECK: upmem.wait_for %[[S1]] {upmem.async}
// CHECK: upmem.sync %[[S1]]

#map = affine_map<(d0) -> (d0 floordiv 256, (d0 floordiv 64) mod 4, 0, 0, 0, 0)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0 * 1024 + d3 * 16 + d5, (d1 * 16 + d2) * 4 + d6)>
#map2 = affine_map<(d0) -> ((d0 mod 256) floordiv 64, d0 mod 64, 0, 0, 0, 0, 0)>
#map3 = affine_map<(d0) -> ((d0 floordiv 64) mod 4, d0 floordiv 256, (d0 mod 64) * 16, 0, 0, 0)>
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>
module {
  memref.global "private" @__cnm_scratch_1 : memref<4x8x4096xi32> {alignment = 64 : i64}
  memref.global "private" @__cnm_scratch_0 : memref<4x8x4096xi32> {alignment = 64 : i64}
  memref.global "private" @__cnm_repack_1 : memref<4x64x16x64x1x16x4xi32>
  memref.global "private" @__cnm_repack_0 : memref<4x64x16x64x1x16x4xi32>
  memref.global "private" constant @__constant_xi32 : memref<i32> = dense<0> {alignment = 64 : i64}
  func.func @par(%arg0: memref<8x4096xi32>, %arg1: memref<4096x4096xi32> {cinm.static}, %arg2: memref<4096x4096xi32> {cinm.static}, %arg3: memref<8x4096xi32>, %arg4: memref<8x4096xi32>) {
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %0 = upmem.alloc_dpus : !upmem.hierarchy<2048x16>
    upmem.load_program @dpu_kernels::@program on %0 : !upmem.hierarchy<2048x16>
    %s1 = upmem.alloc_dpus : !upmem.hierarchy<2048x16>
    upmem.load_program @dpu_kernels::@program on %s1 : !upmem.hierarchy<2048x16>
    %1 = memref.get_global @__cnm_scratch_0 : memref<4x8x4096xi32>
    %expand_shape = memref.expand_shape %arg0 [[0], [1, 2, 3, 4, 5]] output_shape [8, 4, 64, 1, 1, 16] : memref<8x4096xi32> into memref<8x4x64x1x1x16xi32>
    upmem.scatter_on_array %expand_shape[1024 elts, #map] onto @buf_1 of %0 : memref<8x4x64x1x1x16xi32> onto !upmem.hierarchy<2048x16>
    %2 = memref.get_global @__cnm_repack_0 : memref<4x64x16x64x1x16x4xi32> {cinm.static}
    cnm.compact_buffer %arg1 into %2[#map1] {cinm.static} : memref<4096x4096xi32> into memref<4x64x16x64x1x16x4xi32>
    upmem.scatter_on_array %2[65536 elts, #map2] onto @buf_0 slot %c0 of %0 : memref<4x64x16x64x1x16x4xi32> onto !upmem.hierarchy<2048x16>
    upmem.wait_for %0 : !upmem.hierarchy<2048x16>
    %expand_shape_0 = memref.expand_shape %1 [[0], [1], [2, 3, 4, 5]] output_shape [4, 8, 1024, 1, 1, 4] : memref<4x8x4096xi32> into memref<4x8x1024x1x1x4xi32>
    upmem.gather_from_array %expand_shape_0[64 elts, #map3] from @buf of %0 : memref<4x8x1024x1x1x4xi32> from !upmem.hierarchy<2048x16>
    affine.for %i = 0 to 8 {
      affine.for %i_2 = 0 to 4096 {
        %5 = affine.for %i_3 = 0 to 4 iter_args(%acc = %c0_i32) -> (i32) {
          %6 = affine.load %1[%i_3, %i, %i_2] : memref<4x8x4096xi32>
          %7 = arith.addi %6, %acc : i32
          affine.yield %7 : i32
        }
        affine.store %5, %arg3[%i, %i_2] : memref<8x4096xi32>
      }
    }
    %3 = memref.get_global @__cnm_scratch_1 : memref<4x8x4096xi32>
    upmem.scatter_on_array %expand_shape[1024 elts, #map] onto @buf_1 of %s1 : memref<8x4x64x1x1x16xi32> onto !upmem.hierarchy<2048x16>
    %4 = memref.get_global @__cnm_repack_1 : memref<4x64x16x64x1x16x4xi32> {cinm.static}
    cnm.compact_buffer %arg2 into %4[#map1] {cinm.static} : memref<4096x4096xi32> into memref<4x64x16x64x1x16x4xi32>
    upmem.scatter_on_array %4[65536 elts, #map2] onto @buf_0 slot %c1 of %s1 : memref<4x64x16x64x1x16x4xi32> onto !upmem.hierarchy<2048x16>
    upmem.wait_for %s1 : !upmem.hierarchy<2048x16>
    %expand_shape_1 = memref.expand_shape %3 [[0], [1], [2, 3, 4, 5]] output_shape [4, 8, 1024, 1, 1, 4] : memref<4x8x4096xi32> into memref<4x8x1024x1x1x4xi32>
    upmem.gather_from_array %expand_shape_1[64 elts, #map3] from @buf of %s1 : memref<4x8x1024x1x1x4xi32> from !upmem.hierarchy<2048x16>
    affine.for %i = 0 to 8 {
      affine.for %i_2 = 0 to 4096 {
        %5 = affine.for %i_3 = 0 to 4 iter_args(%acc = %c0_i32) -> (i32) {
          %6 = affine.load %3[%i_3, %i, %i_2] : memref<4x8x4096xi32>
          %7 = arith.addi %6, %acc : i32
          affine.yield %7 : i32
        }
        affine.store %5, %arg4[%i, %i_2] : memref<8x4096xi32>
      }
    }
    upmem.free_dpus %0 : !upmem.hierarchy<2048x16>
    upmem.free_dpus %s1 : !upmem.hierarchy<2048x16>
    return
  }
  func.func @seq(%arg0: memref<8x4096xi32>, %arg1: memref<4096x4096xi32> {cinm.static}, %arg2: memref<4096x4096xi32> {cinm.static}, %arg3: memref<8x4096xi32>, %arg4: memref<8x4096xi32>) {
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %0 = upmem.alloc_dpus : !upmem.hierarchy<2048x16>
    upmem.load_program @dpu_kernels::@program on %0 : !upmem.hierarchy<2048x16>
    %s1 = upmem.alloc_dpus : !upmem.hierarchy<2048x16>
    upmem.load_program @dpu_kernels::@program on %s1 : !upmem.hierarchy<2048x16>
    %1 = memref.get_global @__cnm_scratch_0 : memref<4x8x4096xi32>
    %expand_shape = memref.expand_shape %arg0 [[0], [1, 2, 3, 4, 5]] output_shape [8, 4, 64, 1, 1, 16] : memref<8x4096xi32> into memref<8x4x64x1x1x16xi32>
    upmem.scatter_on_array %expand_shape[1024 elts, #map] onto @buf_1 of %0 : memref<8x4x64x1x1x16xi32> onto !upmem.hierarchy<2048x16>
    %2 = memref.get_global @__cnm_repack_0 : memref<4x64x16x64x1x16x4xi32> {cinm.static}
    cnm.compact_buffer %arg1 into %2[#map1] {cinm.static} : memref<4096x4096xi32> into memref<4x64x16x64x1x16x4xi32>
    upmem.scatter_on_array %2[65536 elts, #map2] onto @buf_0 slot %c0 of %0 : memref<4x64x16x64x1x16x4xi32> onto !upmem.hierarchy<2048x16>
    upmem.wait_for %0 : !upmem.hierarchy<2048x16>
    %expand_shape_0 = memref.expand_shape %1 [[0], [1], [2, 3, 4, 5]] output_shape [4, 8, 1024, 1, 1, 4] : memref<4x8x4096xi32> into memref<4x8x1024x1x1x4xi32>
    upmem.gather_from_array %expand_shape_0[64 elts, #map3] from @buf of %0 : memref<4x8x1024x1x1x4xi32> from !upmem.hierarchy<2048x16>
    affine.for %i = 0 to 8 {
      affine.for %i_2 = 0 to 4096 {
        %5 = affine.for %i_3 = 0 to 4 iter_args(%acc = %c0_i32) -> (i32) {
          %6 = affine.load %1[%i_3, %i, %i_2] : memref<4x8x4096xi32>
          %7 = arith.addi %6, %acc : i32
          affine.yield %7 : i32
        }
        affine.store %5, %arg3[%i, %i_2] : memref<8x4096xi32>
      }
    }
    %3 = memref.get_global @__cnm_scratch_1 : memref<4x8x4096xi32>
    %chained = memref.expand_shape %arg3 [[0], [1, 2, 3, 4, 5]] output_shape [8, 4, 64, 1, 1, 16] : memref<8x4096xi32> into memref<8x4x64x1x1x16xi32>
    upmem.scatter_on_array %chained[1024 elts, #map] onto @buf_1 of %s1 : memref<8x4x64x1x1x16xi32> onto !upmem.hierarchy<2048x16>
    %4 = memref.get_global @__cnm_repack_1 : memref<4x64x16x64x1x16x4xi32> {cinm.static}
    cnm.compact_buffer %arg2 into %4[#map1] {cinm.static} : memref<4096x4096xi32> into memref<4x64x16x64x1x16x4xi32>
    upmem.scatter_on_array %4[65536 elts, #map2] onto @buf_0 slot %c1 of %s1 : memref<4x64x16x64x1x16x4xi32> onto !upmem.hierarchy<2048x16>
    upmem.wait_for %s1 : !upmem.hierarchy<2048x16>
    %expand_shape_1 = memref.expand_shape %3 [[0], [1], [2, 3, 4, 5]] output_shape [4, 8, 1024, 1, 1, 4] : memref<4x8x4096xi32> into memref<4x8x1024x1x1x4xi32>
    upmem.gather_from_array %expand_shape_1[64 elts, #map3] from @buf of %s1 : memref<4x8x1024x1x1x4xi32> from !upmem.hierarchy<2048x16>
    affine.for %i = 0 to 8 {
      affine.for %i_2 = 0 to 4096 {
        %5 = affine.for %i_3 = 0 to 4 iter_args(%acc = %c0_i32) -> (i32) {
          %6 = affine.load %3[%i_3, %i, %i_2] : memref<4x8x4096xi32>
          %7 = arith.addi %6, %acc : i32
          affine.yield %7 : i32
        }
        affine.store %5, %arg4[%i, %i_2] : memref<8x4096xi32>
      }
    }
    upmem.free_dpus %0 : !upmem.hierarchy<2048x16>
    upmem.free_dpus %s1 : !upmem.hierarchy<2048x16>
    return
  }
  module @dpu_kernels {
    upmem.dpu_program @program() tasklets(16) {
      %c15 = arith.constant 15 : index
      %c14 = arith.constant 14 : index
      %c13 = arith.constant 13 : index
      %c12 = arith.constant 12 : index
      %c11 = arith.constant 11 : index
      %c10 = arith.constant 10 : index
      %c9 = arith.constant 9 : index
      %c8 = arith.constant 8 : index
      %c7 = arith.constant 7 : index
      %c6 = arith.constant 6 : index
      %c5 = arith.constant 5 : index
      %c4 = arith.constant 4 : index
      %c3 = arith.constant 3 : index
      %c64 = arith.constant 64 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      %c2 = arith.constant 2 : index
      %c0_i32 = arith.constant 0 : i32
      %alloca = memref.alloca() : memref<1x1x16x4xi32, #upmem.wram>
      %alloca_0 = memref.alloca() : memref<1x1x1x16xi32, #upmem.wram>
      %mram_buf = upmem.static_alloc @buf(mram) noinit : memref<16x1x1x4xi32, #upmem.mram>
      %mram_buf_1 = upmem.static_alloc @buf_0(mram) noinit slots 2 : memref<2x16x64x1x16x4xi32, #upmem.mram>
      %mram_buf_2 = upmem.static_alloc @buf_1(mram) noinit : memref<64x1x1x16xi32, #upmem.mram>
      %wram_buf = upmem.static_alloc @launch_count(wram) zeroinit : memref<16xi32, #upmem.wram>
      %0 = upmem.tasklet_dim()
      %1 = memref.load %wram_buf[%0] : memref<16xi32, #upmem.wram>
      %2 = arith.addi %1, %c1_i32 : i32
      memref.store %2, %wram_buf[%0] : memref<16xi32, #upmem.wram>
      %3 = arith.index_cast %1 : i32 to index
      %4 = arith.remui %3, %c2 : index
      %subview = memref.subview %mram_buf[%0, 0, 0, 0] [1, 1, 1, 4] [1, 1, 1, 1] : memref<16x1x1x4xi32, #upmem.mram> to memref<1x1x4xi32, strided<[4, 4, 1], offset: ?>, #upmem.mram>
      %alloca_3 = memref.alloca() : memref<1x1x4xi32, #upmem.wram>
      memref.store %c0_i32, %alloca_3[%c0, %c0, %c0] : memref<1x1x4xi32, #upmem.wram>
      memref.store %c0_i32, %alloca_3[%c0, %c0, %c1] : memref<1x1x4xi32, #upmem.wram>
      %5:2 = scf.for %arg0 = %c0 to %c64 step %c1 iter_args(%arg1 = %c0_i32, %arg2 = %c0_i32) -> (i32, i32) {
        %subview_4 = memref.subview %mram_buf_2[%arg0, 0, 0, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<64x1x1x16xi32, #upmem.mram> to memref<1x1x1x16xi32, strided<[16, 16, 16, 1], offset: ?>, #upmem.mram>
        %subview_5 = memref.subview %mram_buf_1[%4, %0, %arg0, 0, 0, 0] [1, 1, 1, 1, 16, 4] [1, 1, 1, 1, 1, 1] : memref<2x16x64x1x16x4xi32, #upmem.mram> to memref<1x1x16x4xi32, strided<[64, 64, 4, 1], offset: ?>, #upmem.mram>
        upmem.local_transfer %subview_4 into %alloca_0 : memref<1x1x1x16xi32, strided<[16, 16, 16, 1], offset: ?>, #upmem.mram> to memref<1x1x1x16xi32, #upmem.wram>
        upmem.local_transfer %subview_5 into %alloca : memref<1x1x16x4xi32, strided<[64, 64, 4, 1], offset: ?>, #upmem.mram> to memref<1x1x16x4xi32, #upmem.wram>
        %6 = memref.load %alloca_3[%c0, %c0, %c0] : memref<1x1x4xi32, #upmem.wram>
        %7 = memref.load %alloca_0[%c0, %c0, %c0, %c0] : memref<1x1x1x16xi32, #upmem.wram>
        %8 = memref.load %alloca[%c0, %c0, %c0, %c0] : memref<1x1x16x4xi32, #upmem.wram>
        %9 = arith.muli %7, %8 : i32
        %10 = arith.addi %6, %9 : i32
        %11 = memref.load %alloca_0[%c0, %c0, %c0, %c1] : memref<1x1x1x16xi32, #upmem.wram>
        %12 = memref.load %alloca[%c0, %c0, %c1, %c0] : memref<1x1x16x4xi32, #upmem.wram>
        %13 = arith.muli %11, %12 : i32
        %14 = arith.addi %10, %13 : i32
        %15 = memref.load %alloca_0[%c0, %c0, %c0, %c2] : memref<1x1x1x16xi32, #upmem.wram>
        %16 = memref.load %alloca[%c0, %c0, %c2, %c0] : memref<1x1x16x4xi32, #upmem.wram>
        %17 = arith.muli %15, %16 : i32
        %18 = arith.addi %14, %17 : i32
        %19 = memref.load %alloca_0[%c0, %c0, %c0, %c3] : memref<1x1x1x16xi32, #upmem.wram>
        %20 = memref.load %alloca[%c0, %c0, %c3, %c0] : memref<1x1x16x4xi32, #upmem.wram>
        %21 = arith.muli %19, %20 : i32
        %22 = arith.addi %18, %21 : i32
        %23 = memref.load %alloca_0[%c0, %c0, %c0, %c4] : memref<1x1x1x16xi32, #upmem.wram>
        %24 = memref.load %alloca[%c0, %c0, %c4, %c0] : memref<1x1x16x4xi32, #upmem.wram>
        %25 = arith.muli %23, %24 : i32
        %26 = arith.addi %22, %25 : i32
        %27 = memref.load %alloca_0[%c0, %c0, %c0, %c5] : memref<1x1x1x16xi32, #upmem.wram>
        %28 = memref.load %alloca[%c0, %c0, %c5, %c0] : memref<1x1x16x4xi32, #upmem.wram>
        %29 = arith.muli %27, %28 : i32
        %30 = arith.addi %26, %29 : i32
        %31 = memref.load %alloca_0[%c0, %c0, %c0, %c6] : memref<1x1x1x16xi32, #upmem.wram>
        %32 = memref.load %alloca[%c0, %c0, %c6, %c0] : memref<1x1x16x4xi32, #upmem.wram>
        %33 = arith.muli %31, %32 : i32
        %34 = arith.addi %30, %33 : i32
        %35 = memref.load %alloca_0[%c0, %c0, %c0, %c7] : memref<1x1x1x16xi32, #upmem.wram>
        %36 = memref.load %alloca[%c0, %c0, %c7, %c0] : memref<1x1x16x4xi32, #upmem.wram>
        %37 = arith.muli %35, %36 : i32
        %38 = arith.addi %34, %37 : i32
        %39 = memref.load %alloca_0[%c0, %c0, %c0, %c8] : memref<1x1x1x16xi32, #upmem.wram>
        %40 = memref.load %alloca[%c0, %c0, %c8, %c0] : memref<1x1x16x4xi32, #upmem.wram>
        %41 = arith.muli %39, %40 : i32
        %42 = arith.addi %38, %41 : i32
        %43 = memref.load %alloca_0[%c0, %c0, %c0, %c9] : memref<1x1x1x16xi32, #upmem.wram>
        %44 = memref.load %alloca[%c0, %c0, %c9, %c0] : memref<1x1x16x4xi32, #upmem.wram>
        %45 = arith.muli %43, %44 : i32
        %46 = arith.addi %42, %45 : i32
        %47 = memref.load %alloca_0[%c0, %c0, %c0, %c10] : memref<1x1x1x16xi32, #upmem.wram>
        %48 = memref.load %alloca[%c0, %c0, %c10, %c0] : memref<1x1x16x4xi32, #upmem.wram>
        %49 = arith.muli %47, %48 : i32
        %50 = arith.addi %46, %49 : i32
        %51 = memref.load %alloca_0[%c0, %c0, %c0, %c11] : memref<1x1x1x16xi32, #upmem.wram>
        %52 = memref.load %alloca[%c0, %c0, %c11, %c0] : memref<1x1x16x4xi32, #upmem.wram>
        %53 = arith.muli %51, %52 : i32
        %54 = arith.addi %50, %53 : i32
        %55 = memref.load %alloca_0[%c0, %c0, %c0, %c12] : memref<1x1x1x16xi32, #upmem.wram>
        %56 = memref.load %alloca[%c0, %c0, %c12, %c0] : memref<1x1x16x4xi32, #upmem.wram>
        %57 = arith.muli %55, %56 : i32
        %58 = arith.addi %54, %57 : i32
        %59 = memref.load %alloca_0[%c0, %c0, %c0, %c13] : memref<1x1x1x16xi32, #upmem.wram>
        %60 = memref.load %alloca[%c0, %c0, %c13, %c0] : memref<1x1x16x4xi32, #upmem.wram>
        %61 = arith.muli %59, %60 : i32
        %62 = arith.addi %58, %61 : i32
        %63 = memref.load %alloca_0[%c0, %c0, %c0, %c14] : memref<1x1x1x16xi32, #upmem.wram>
        %64 = memref.load %alloca[%c0, %c0, %c14, %c0] : memref<1x1x16x4xi32, #upmem.wram>
        %65 = arith.muli %63, %64 : i32
        %66 = arith.addi %62, %65 : i32
        %67 = memref.load %alloca_0[%c0, %c0, %c0, %c15] : memref<1x1x1x16xi32, #upmem.wram>
        %68 = memref.load %alloca[%c0, %c0, %c15, %c0] : memref<1x1x16x4xi32, #upmem.wram>
        %69 = arith.muli %67, %68 : i32
        %70 = arith.addi %66, %69 : i32
        memref.store %70, %alloca_3[%c0, %c0, %c0] : memref<1x1x4xi32, #upmem.wram>
        %71 = memref.load %alloca_3[%c0, %c0, %c1] : memref<1x1x4xi32, #upmem.wram>
        %72 = memref.load %alloca[%c0, %c0, %c0, %c1] : memref<1x1x16x4xi32, #upmem.wram>
        %73 = arith.muli %7, %72 : i32
        %74 = arith.addi %71, %73 : i32
        %75 = memref.load %alloca[%c0, %c0, %c1, %c1] : memref<1x1x16x4xi32, #upmem.wram>
        %76 = arith.muli %11, %75 : i32
        %77 = arith.addi %74, %76 : i32
        %78 = memref.load %alloca[%c0, %c0, %c2, %c1] : memref<1x1x16x4xi32, #upmem.wram>
        %79 = arith.muli %15, %78 : i32
        %80 = arith.addi %77, %79 : i32
        %81 = memref.load %alloca[%c0, %c0, %c3, %c1] : memref<1x1x16x4xi32, #upmem.wram>
        %82 = arith.muli %19, %81 : i32
        %83 = arith.addi %80, %82 : i32
        %84 = memref.load %alloca[%c0, %c0, %c4, %c1] : memref<1x1x16x4xi32, #upmem.wram>
        %85 = arith.muli %23, %84 : i32
        %86 = arith.addi %83, %85 : i32
        %87 = memref.load %alloca[%c0, %c0, %c5, %c1] : memref<1x1x16x4xi32, #upmem.wram>
        %88 = arith.muli %27, %87 : i32
        %89 = arith.addi %86, %88 : i32
        %90 = memref.load %alloca[%c0, %c0, %c6, %c1] : memref<1x1x16x4xi32, #upmem.wram>
        %91 = arith.muli %31, %90 : i32
        %92 = arith.addi %89, %91 : i32
        %93 = memref.load %alloca[%c0, %c0, %c7, %c1] : memref<1x1x16x4xi32, #upmem.wram>
        %94 = arith.muli %35, %93 : i32
        %95 = arith.addi %92, %94 : i32
        %96 = memref.load %alloca[%c0, %c0, %c8, %c1] : memref<1x1x16x4xi32, #upmem.wram>
        %97 = arith.muli %39, %96 : i32
        %98 = arith.addi %95, %97 : i32
        %99 = memref.load %alloca[%c0, %c0, %c9, %c1] : memref<1x1x16x4xi32, #upmem.wram>
        %100 = arith.muli %43, %99 : i32
        %101 = arith.addi %98, %100 : i32
        %102 = memref.load %alloca[%c0, %c0, %c10, %c1] : memref<1x1x16x4xi32, #upmem.wram>
        %103 = arith.muli %47, %102 : i32
        %104 = arith.addi %101, %103 : i32
        %105 = memref.load %alloca[%c0, %c0, %c11, %c1] : memref<1x1x16x4xi32, #upmem.wram>
        %106 = arith.muli %51, %105 : i32
        %107 = arith.addi %104, %106 : i32
        %108 = memref.load %alloca[%c0, %c0, %c12, %c1] : memref<1x1x16x4xi32, #upmem.wram>
        %109 = arith.muli %55, %108 : i32
        %110 = arith.addi %107, %109 : i32
        %111 = memref.load %alloca[%c0, %c0, %c13, %c1] : memref<1x1x16x4xi32, #upmem.wram>
        %112 = arith.muli %59, %111 : i32
        %113 = arith.addi %110, %112 : i32
        %114 = memref.load %alloca[%c0, %c0, %c14, %c1] : memref<1x1x16x4xi32, #upmem.wram>
        %115 = arith.muli %63, %114 : i32
        %116 = arith.addi %113, %115 : i32
        %117 = memref.load %alloca[%c0, %c0, %c15, %c1] : memref<1x1x16x4xi32, #upmem.wram>
        %118 = arith.muli %67, %117 : i32
        %119 = arith.addi %116, %118 : i32
        memref.store %119, %alloca_3[%c0, %c0, %c1] : memref<1x1x4xi32, #upmem.wram>
        %120 = memref.load %alloca[%c0, %c0, %c0, %c2] : memref<1x1x16x4xi32, #upmem.wram>
        %121 = arith.muli %7, %120 : i32
        %122 = arith.addi %arg2, %121 : i32
        %123 = memref.load %alloca[%c0, %c0, %c1, %c2] : memref<1x1x16x4xi32, #upmem.wram>
        %124 = arith.muli %11, %123 : i32
        %125 = arith.addi %122, %124 : i32
        %126 = memref.load %alloca[%c0, %c0, %c2, %c2] : memref<1x1x16x4xi32, #upmem.wram>
        %127 = arith.muli %15, %126 : i32
        %128 = arith.addi %125, %127 : i32
        %129 = memref.load %alloca[%c0, %c0, %c3, %c2] : memref<1x1x16x4xi32, #upmem.wram>
        %130 = arith.muli %19, %129 : i32
        %131 = arith.addi %128, %130 : i32
        %132 = memref.load %alloca[%c0, %c0, %c4, %c2] : memref<1x1x16x4xi32, #upmem.wram>
        %133 = arith.muli %23, %132 : i32
        %134 = arith.addi %131, %133 : i32
        %135 = memref.load %alloca[%c0, %c0, %c5, %c2] : memref<1x1x16x4xi32, #upmem.wram>
        %136 = arith.muli %27, %135 : i32
        %137 = arith.addi %134, %136 : i32
        %138 = memref.load %alloca[%c0, %c0, %c6, %c2] : memref<1x1x16x4xi32, #upmem.wram>
        %139 = arith.muli %31, %138 : i32
        %140 = arith.addi %137, %139 : i32
        %141 = memref.load %alloca[%c0, %c0, %c7, %c2] : memref<1x1x16x4xi32, #upmem.wram>
        %142 = arith.muli %35, %141 : i32
        %143 = arith.addi %140, %142 : i32
        %144 = memref.load %alloca[%c0, %c0, %c8, %c2] : memref<1x1x16x4xi32, #upmem.wram>
        %145 = arith.muli %39, %144 : i32
        %146 = arith.addi %143, %145 : i32
        %147 = memref.load %alloca[%c0, %c0, %c9, %c2] : memref<1x1x16x4xi32, #upmem.wram>
        %148 = arith.muli %43, %147 : i32
        %149 = arith.addi %146, %148 : i32
        %150 = memref.load %alloca[%c0, %c0, %c10, %c2] : memref<1x1x16x4xi32, #upmem.wram>
        %151 = arith.muli %47, %150 : i32
        %152 = arith.addi %149, %151 : i32
        %153 = memref.load %alloca[%c0, %c0, %c11, %c2] : memref<1x1x16x4xi32, #upmem.wram>
        %154 = arith.muli %51, %153 : i32
        %155 = arith.addi %152, %154 : i32
        %156 = memref.load %alloca[%c0, %c0, %c12, %c2] : memref<1x1x16x4xi32, #upmem.wram>
        %157 = arith.muli %55, %156 : i32
        %158 = arith.addi %155, %157 : i32
        %159 = memref.load %alloca[%c0, %c0, %c13, %c2] : memref<1x1x16x4xi32, #upmem.wram>
        %160 = arith.muli %59, %159 : i32
        %161 = arith.addi %158, %160 : i32
        %162 = memref.load %alloca[%c0, %c0, %c14, %c2] : memref<1x1x16x4xi32, #upmem.wram>
        %163 = arith.muli %63, %162 : i32
        %164 = arith.addi %161, %163 : i32
        %165 = memref.load %alloca[%c0, %c0, %c15, %c2] : memref<1x1x16x4xi32, #upmem.wram>
        %166 = arith.muli %67, %165 : i32
        %167 = arith.addi %164, %166 : i32
        %168 = memref.load %alloca[%c0, %c0, %c0, %c3] : memref<1x1x16x4xi32, #upmem.wram>
        %169 = arith.muli %7, %168 : i32
        %170 = arith.addi %arg1, %169 : i32
        %171 = memref.load %alloca[%c0, %c0, %c1, %c3] : memref<1x1x16x4xi32, #upmem.wram>
        %172 = arith.muli %11, %171 : i32
        %173 = arith.addi %170, %172 : i32
        %174 = memref.load %alloca[%c0, %c0, %c2, %c3] : memref<1x1x16x4xi32, #upmem.wram>
        %175 = arith.muli %15, %174 : i32
        %176 = arith.addi %173, %175 : i32
        %177 = memref.load %alloca[%c0, %c0, %c3, %c3] : memref<1x1x16x4xi32, #upmem.wram>
        %178 = arith.muli %19, %177 : i32
        %179 = arith.addi %176, %178 : i32
        %180 = memref.load %alloca[%c0, %c0, %c4, %c3] : memref<1x1x16x4xi32, #upmem.wram>
        %181 = arith.muli %23, %180 : i32
        %182 = arith.addi %179, %181 : i32
        %183 = memref.load %alloca[%c0, %c0, %c5, %c3] : memref<1x1x16x4xi32, #upmem.wram>
        %184 = arith.muli %27, %183 : i32
        %185 = arith.addi %182, %184 : i32
        %186 = memref.load %alloca[%c0, %c0, %c6, %c3] : memref<1x1x16x4xi32, #upmem.wram>
        %187 = arith.muli %31, %186 : i32
        %188 = arith.addi %185, %187 : i32
        %189 = memref.load %alloca[%c0, %c0, %c7, %c3] : memref<1x1x16x4xi32, #upmem.wram>
        %190 = arith.muli %35, %189 : i32
        %191 = arith.addi %188, %190 : i32
        %192 = memref.load %alloca[%c0, %c0, %c8, %c3] : memref<1x1x16x4xi32, #upmem.wram>
        %193 = arith.muli %39, %192 : i32
        %194 = arith.addi %191, %193 : i32
        %195 = memref.load %alloca[%c0, %c0, %c9, %c3] : memref<1x1x16x4xi32, #upmem.wram>
        %196 = arith.muli %43, %195 : i32
        %197 = arith.addi %194, %196 : i32
        %198 = memref.load %alloca[%c0, %c0, %c10, %c3] : memref<1x1x16x4xi32, #upmem.wram>
        %199 = arith.muli %47, %198 : i32
        %200 = arith.addi %197, %199 : i32
        %201 = memref.load %alloca[%c0, %c0, %c11, %c3] : memref<1x1x16x4xi32, #upmem.wram>
        %202 = arith.muli %51, %201 : i32
        %203 = arith.addi %200, %202 : i32
        %204 = memref.load %alloca[%c0, %c0, %c12, %c3] : memref<1x1x16x4xi32, #upmem.wram>
        %205 = arith.muli %55, %204 : i32
        %206 = arith.addi %203, %205 : i32
        %207 = memref.load %alloca[%c0, %c0, %c13, %c3] : memref<1x1x16x4xi32, #upmem.wram>
        %208 = arith.muli %59, %207 : i32
        %209 = arith.addi %206, %208 : i32
        %210 = memref.load %alloca[%c0, %c0, %c14, %c3] : memref<1x1x16x4xi32, #upmem.wram>
        %211 = arith.muli %63, %210 : i32
        %212 = arith.addi %209, %211 : i32
        %213 = memref.load %alloca[%c0, %c0, %c15, %c3] : memref<1x1x16x4xi32, #upmem.wram>
        %214 = arith.muli %67, %213 : i32
        %215 = arith.addi %212, %214 : i32
        scf.yield %215, %167 : i32, i32
      } {upmem.nounroll}
      memref.store %5#1, %alloca_3[%c0, %c0, %c2] : memref<1x1x4xi32, #upmem.wram>
      memref.store %5#0, %alloca_3[%c0, %c0, %c3] : memref<1x1x4xi32, #upmem.wram>
      upmem.local_transfer %alloca_3 into %subview : memref<1x1x4xi32, #upmem.wram> to memref<1x1x4xi32, strided<[4, 4, 1], offset: ?>, #upmem.mram>
      upmem.return
    }
  }
}
