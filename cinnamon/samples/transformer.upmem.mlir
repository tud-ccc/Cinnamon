#map = affine_map<(d0, d1, d2) -> (d1 * 8 + d2)>
#map1 = affine_map<(d0, d1, d2) -> (0)>
#map2 = affine_map<(d0, d1, d2) -> (d1 * 8 + d2, 0)>
#map3 = affine_map<(d0, d1, d2) -> (d1 * 16 + d2)>
#map4 = affine_map<(d0, d1, d2) -> (d1 * 16 + d2, 0)>
#map5 = affine_map<(d0, d1, d2) -> (d2)>
#map6 = affine_map<(d0, d1, d2) -> (d0 * 1024 + d1 * 16 + d2)>
module {
  memref.global "private" constant @__tconstant_1xi64 : memref<1xi64> = dense<1024> {alignment = 64 : i64}
  memref.global "private" constant @__tconstant_256x4xi32 : memref<256x4xi32> = dense<0> {alignment = 64 : i64}
  memref.global "private" constant @__tconstant_2xi64 : memref<2xi64> = dense<[256, 4]> {alignment = 64 : i64}
  memref.global "private" constant @__tconstant_1024xi32 : memref<1024xi32> = dense<0> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1xi64_7 : memref<1xi64> = dense<32768>
  memref.global "private" constant @__constant_256x1xf32 : memref<256x1xf32> = dense<0.000000e+00>
  memref.global "private" constant @__constant_2xi64_8 : memref<2xi64> = dense<[768, 1]>
  memref.global "private" constant @__constant_1xi64_6 : memref<1xi64> = dense<768>
  memref.global "private" constant @__constant_48x6xf32 : memref<48x6xf32> = dense<0.000000e+00>
  memref.global "private" constant @__constant_2xi64_7 : memref<2xi64> = dense<[48, 6]>
  memref.global "private" constant @__constant_48x1xf32 : memref<48x1xf32> = dense<0.000000e+00>
  memref.global "private" constant @__constant_2xi64_6 : memref<2xi64> = dense<[288, 1]>
  memref.global "private" constant @__constant_1xi64_5 : memref<1xi64> = dense<1024>
  memref.global "private" constant @__constant_256x4xf32 : memref<256x4xf32> = dense<0.000000e+00>
  memref.global "private" constant @__constant_2xi64_5 : memref<2xi64> = dense<[256, 4]>
  memref.global "private" constant @__constant_1024xf32 : memref<1024xf32> = dense<0.000000e+00>
  memref.global "private" constant @__constant_1xi64_4 : memref<1xi64> = dense<4096>
  memref.global "private" constant @__constant_256x16xf32 : memref<256x16xf32> = dense<0.000000e+00>
  memref.global "private" constant @__constant_2xi64_4 : memref<2xi64> = dense<[256, 16]>
  memref.global "private" constant @__constant_4096xf32 : memref<4096xf32> = dense<0.000000e+00>
  memref.global "private" constant @__constant_1xi64_3 : memref<1xi64> = dense<48>
  memref.global "private" constant @__constant_8x6xf32 : memref<8x6xf32> = dense<0.000000e+00>
  memref.global "private" constant @__constant_2xi64_3 : memref<2xi64> = dense<[8, 6]>
  memref.global "private" constant @__constant_48xf32 : memref<48xf32> = dense<0.000000e+00>
  memref.global "private" constant @__constant_1xi64_2 : memref<1xi64> = dense<256>
  memref.global "private" constant @__constant_128x2xf32 : memref<128x2xf32> = dense<0.000000e+00>
  memref.global "private" constant @__constant_2xi64_2 : memref<2xi64> = dense<[128, 2]>
  memref.global "private" constant @__constant_256xf32 : memref<256xf32> = dense<0.000000e+00>
  memref.global "private" constant @__constant_1xi64_1 : memref<1xi64> = dense<288>
  memref.global "private" constant @__constant_16x18xf32 : memref<16x18xf32> = dense<0.000000e+00>
  memref.global "private" constant @__constant_2xi64_1 : memref<2xi64> = dense<[16, 18]>
  memref.global "private" constant @__constant_288xf32 : memref<288xf32> = dense<0.000000e+00>
  memref.global "private" constant @__constant_xf32_0 : memref<f32> = dense<0.000000e+00>
  memref.global "private" constant @__constant_1xi64_0 : memref<1xi64> = dense<262144>
  memref.global "private" constant @__constant_4096x64xf32 : memref<4096x64xf32> = dense<0.000000e+00>
  memref.global "private" constant @__constant_4096x2xf32 : memref<4096x2xf32> = dense<0.000000e+00>
  memref.global "private" constant @__constant_2xi64_0 : memref<2xi64> = dense<[4096, 64]>
  memref.global "private" constant @__constant_262144xf32 : memref<262144xf32> = dense<0.000000e+00>
  memref.global "private" constant @__constant_xf32 : memref<f32> = dense<0xFF800000>
  memref.global "private" constant @__constant_1xi64 : memref<1xi64> = dense<1048576>
  memref.global "private" constant @__constant_4096x256xf32 : memref<4096x256xf32> = dense<0.000000e+00>
  memref.global "private" constant @__constant_2xi64 : memref<2xi64> = dense<[4096, 256]>
  memref.global "private" constant @__constant_1048576xf32 : memref<1048576xf32> = dense<0.000000e+00>
  func.func @forward(%arg0: index, %arg1: index, %arg2: memref<6x256x288xf32>, %arg3: memref<6x256x288xf32>, %arg4: memref<32000x288xf32>, %arg5: memref<6x288xf32>, %arg6: memref<6x288x288xf32>, %arg7: memref<6x288x288xf32>, %arg8: memref<6x288x288xf32>, %arg9: memref<6x288x288xf32>, %arg10: memref<6x768x288xf32>, %arg11: memref<6x288x768xf32>, %arg12: memref<6x768x288xf32>, %arg13: memref<6x288xf32>, %arg14: memref<288xf32>, %arg15: memref<32000x288xf32>) -> memref<32000xf32> {
    %c256 = arith.constant 256 : index
    %c32768 = arith.constant 32768 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c6 = arith.constant 6 : index
    %c48 = arith.constant 48 : index
    %c288 = arith.constant 288 : index
    %c768 = arith.constant 768 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 4.800000e+01 : f32
    %cst_2 = arith.constant 1.000000e+04 : f32
    %subview = memref.subview %arg4[%arg0, 0] [1, 288] [1, 1] : memref<32000x288xf32> to memref<288xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<288xf32>
    memref.copy %subview, %alloc : memref<288xf32, strided<[1], offset: ?>> to memref<288xf32>
    %alloc_3 = memref.alloc() : memref<288xf32>
    %alloc_4 = memref.alloc() : memref<288xf32>
    %alloc_5 = memref.alloc() : memref<288x1xf32>
    %alloc_6 = memref.alloc() : memref<288x1xf32>
    %alloc_7 = memref.alloc() : memref<1x288xf32>
    %alloc_8 = memref.alloc() : memref<48x1xf32>
    %alloc_9 = memref.alloc() : memref<288x1xf32>
    %alloc_10 = memref.alloc() : memref<288x1xf32>
    %alloc_11 = memref.alloc() : memref<1x288xf32>
    %alloc_12 = memref.alloc() : memref<48x1xf32>
    %alloc_13 = memref.alloc() : memref<288x1xf32>
    %alloc_14 = memref.alloc() : memref<288x1xf32>
    %alloc_15 = memref.alloc() : memref<1x288xf32>
    %alloc_16 = memref.alloc() : memref<48x1xf32>
    %alloc_17 = memref.alloc() : memref<288xf32>
    %alloc_18 = memref.alloc() : memref<288xf32>
    %alloc_19 = memref.alloc() : memref<288xf32>
    %alloc_20 = memref.alloc() : memref<256x288xf32>
    %alloc_21 = memref.alloc() : memref<256x288xf32>
    %alloc_22 = memref.alloc() : memref<288x1xf32>
    %alloc_23 = memref.alloc() : memref<288x1xf32>
    %alloc_24 = memref.alloc() : memref<1x288xf32>
    %alloc_25 = memref.alloc() : memref<48x1xf32>
    %alloc_26 = memref.alloc() : memref<48x6xf32>
    %alloc_27 = memref.alloc() : memref<288xf32>
    %alloc_28 = memref.alloc() : memref<288xf32>
    %alloc_29 = memref.alloc() : memref<768x1xf32>
    %alloc_30 = memref.alloc() : memref<768x1xf32>
    %alloc_31 = memref.alloc() : memref<1x288xf32>
    %alloc_32 = memref.alloc() : memref<48x1xf32>
    %alloc_33 = memref.alloc() : memref<768x1xf32>
    %alloc_34 = memref.alloc() : memref<768x1xf32>
    %alloc_35 = memref.alloc() : memref<1x288xf32>
    %alloc_36 = memref.alloc() : memref<48x1xf32>
    %alloc_37 = memref.alloc() : memref<768xf32>
    %alloc_38 = memref.alloc() : memref<288x1xf32>
    %alloc_39 = memref.alloc() : memref<288x1xf32>
    %alloc_40 = memref.alloc() : memref<1x768xf32>
    %alloc_41 = memref.alloc() : memref<48x1xf32>
    %0 = scf.for %arg16 = %c0 to %c6 step %c1 iter_args(%arg17 = %alloc) -> (memref<288xf32>) {
      %subview_52 = memref.subview %arg5[%arg16, 0] [1, 288] [1, 1] : memref<6x288xf32> to memref<288xf32, strided<[1], offset: ?>>
      memref.copy %arg17, %alloc_3 : memref<288xf32> to memref<288xf32>
      memref.copy %subview_52, %alloc_4 : memref<288xf32, strided<[1], offset: ?>> to memref<288xf32>
      %5 = func.call @rmsnorm(%alloc_3, %alloc_4) : (memref<288xf32>, memref<288xf32>) -> memref<288xf32>
      %subview_53 = memref.subview %arg6[%arg16, 0, 0] [1, 288, 288] [1, 1, 1] : memref<6x288x288xf32> to memref<288x288xf32, strided<[288, 1], offset: ?>>
      %subview_54 = memref.subview %arg7[%arg16, 0, 0] [1, 288, 288] [1, 1, 1] : memref<6x288x288xf32> to memref<288x288xf32, strided<[288, 1], offset: ?>>
      %subview_55 = memref.subview %arg8[%arg16, 0, 0] [1, 288, 288] [1, 1, 1] : memref<6x288x288xf32> to memref<288x288xf32, strided<[288, 1], offset: ?>>
      %6 = memref.get_global @__constant_2xi64_6 : memref<2xi64>
      %reshape_56 = memref.reshape %5(%6) : (memref<288xf32>, memref<2xi64>) -> memref<288x1xf32>
      memref.copy %alloc_5, %alloc_6 : memref<288x1xf32> to memref<288x1xf32>
      %7 = scf.for %arg18 = %c0 to %c288 step %c48 iter_args(%arg19 = %alloc_6) -> (memref<288x1xf32>) {
        %27 = memref.get_global @__constant_48x1xf32 : memref<48x1xf32>
        %subview_82 = memref.subview %subview_53[%arg18, 0] [48, 288] [1, 1] : memref<288x288xf32, strided<[288, 1], offset: ?>> to memref<48x288xf32, strided<[288, 1], offset: ?>>
        %28 = upmem.alloc_dpus : !upmem.hierarchy<1x6x8>
        scf.for %arg20 = %c0 to %c288 step %c1 {
          %29 = memref.load %reshape_56[%arg20, %c0] : memref<288x1xf32>
          memref.store %29, %alloc_7[%c0, %arg20] : memref<1x288xf32>
        }
        upmem.scatter %subview_82[0, 2304, #map] onto %28 : memref<48x288xf32, strided<[288, 1], offset: ?>> onto !upmem.hierarchy<1x6x8>
        upmem.scatter %alloc_7[9216, 2304, #map1] onto %28 : memref<1x288xf32> onto !upmem.hierarchy<1x6x8>
        upmem.scatter %27[18432, 8, #map2] onto %28 : memref<48x1xf32> onto !upmem.hierarchy<1x6x8>
        upmem.launch_func  @dpu_kernels::@forward %28 : !upmem.hierarchy<1x6x8> 
        upmem.gather %alloc_8[18432, 8, #map2] from %28 : memref<48x1xf32> from !upmem.hierarchy<1x6x8>
        upmem.free_dpus %28 : !upmem.hierarchy<1x6x8>
        %alloc_83 = memref.alloc() : memref<288x1xf32>
        memref.copy %arg19, %alloc_83 : memref<288x1xf32> to memref<288x1xf32>
        %subview_84 = memref.subview %alloc_83[%arg18, 0] [48, 1] [1, 1] : memref<288x1xf32> to memref<48x1xf32, strided<[1, 1], offset: ?>>
        memref.copy %alloc_8, %subview_84 : memref<48x1xf32> to memref<48x1xf32, strided<[1, 1], offset: ?>>
        scf.yield %alloc_83 : memref<288x1xf32>
      }
      %8 = memref.get_global @__constant_1xi64_1 : memref<1xi64>
      %reshape_57 = memref.reshape %7(%8) : (memref<288x1xf32>, memref<1xi64>) -> memref<288xf32>
      memref.copy %alloc_9, %alloc_10 : memref<288x1xf32> to memref<288x1xf32>
      %9 = scf.for %arg18 = %c0 to %c288 step %c48 iter_args(%arg19 = %alloc_10) -> (memref<288x1xf32>) {
        %27 = memref.get_global @__constant_48x1xf32 : memref<48x1xf32>
        %subview_82 = memref.subview %subview_54[%arg18, 0] [48, 288] [1, 1] : memref<288x288xf32, strided<[288, 1], offset: ?>> to memref<48x288xf32, strided<[288, 1], offset: ?>>
        %28 = upmem.alloc_dpus : !upmem.hierarchy<1x6x8>
        scf.for %arg20 = %c0 to %c288 step %c1 {
          %29 = memref.load %reshape_56[%arg20, %c0] : memref<288x1xf32>
          memref.store %29, %alloc_11[%c0, %arg20] : memref<1x288xf32>
        }
        upmem.scatter %subview_82[0, 2304, #map] onto %28 : memref<48x288xf32, strided<[288, 1], offset: ?>> onto !upmem.hierarchy<1x6x8>
        upmem.scatter %alloc_11[9216, 2304, #map1] onto %28 : memref<1x288xf32> onto !upmem.hierarchy<1x6x8>
        upmem.scatter %27[18432, 8, #map2] onto %28 : memref<48x1xf32> onto !upmem.hierarchy<1x6x8>
        upmem.launch_func  @dpu_kernels::@forward %28 : !upmem.hierarchy<1x6x8> 
        upmem.gather %alloc_12[18432, 8, #map2] from %28 : memref<48x1xf32> from !upmem.hierarchy<1x6x8>
        upmem.free_dpus %28 : !upmem.hierarchy<1x6x8>
        %alloc_83 = memref.alloc() : memref<288x1xf32>
        memref.copy %arg19, %alloc_83 : memref<288x1xf32> to memref<288x1xf32>
        %subview_84 = memref.subview %alloc_83[%arg18, 0] [48, 1] [1, 1] : memref<288x1xf32> to memref<48x1xf32, strided<[1, 1], offset: ?>>
        memref.copy %alloc_12, %subview_84 : memref<48x1xf32> to memref<48x1xf32, strided<[1, 1], offset: ?>>
        scf.yield %alloc_83 : memref<288x1xf32>
      }
      %reshape_58 = memref.reshape %9(%8) : (memref<288x1xf32>, memref<1xi64>) -> memref<288xf32>
      memref.copy %alloc_13, %alloc_14 : memref<288x1xf32> to memref<288x1xf32>
      %10 = scf.for %arg18 = %c0 to %c288 step %c48 iter_args(%arg19 = %alloc_14) -> (memref<288x1xf32>) {
        %27 = memref.get_global @__constant_48x1xf32 : memref<48x1xf32>
        %subview_82 = memref.subview %subview_55[%arg18, 0] [48, 288] [1, 1] : memref<288x288xf32, strided<[288, 1], offset: ?>> to memref<48x288xf32, strided<[288, 1], offset: ?>>
        %28 = upmem.alloc_dpus : !upmem.hierarchy<1x6x8>
        scf.for %arg20 = %c0 to %c288 step %c1 {
          %29 = memref.load %reshape_56[%arg20, %c0] : memref<288x1xf32>
          memref.store %29, %alloc_15[%c0, %arg20] : memref<1x288xf32>
        }
        upmem.scatter %subview_82[0, 2304, #map] onto %28 : memref<48x288xf32, strided<[288, 1], offset: ?>> onto !upmem.hierarchy<1x6x8>
        upmem.scatter %alloc_15[9216, 2304, #map1] onto %28 : memref<1x288xf32> onto !upmem.hierarchy<1x6x8>
        upmem.scatter %27[18432, 8, #map2] onto %28 : memref<48x1xf32> onto !upmem.hierarchy<1x6x8>
        upmem.launch_func  @dpu_kernels::@forward %28 : !upmem.hierarchy<1x6x8> 
        upmem.gather %alloc_16[18432, 8, #map2] from %28 : memref<48x1xf32> from !upmem.hierarchy<1x6x8>
        upmem.free_dpus %28 : !upmem.hierarchy<1x6x8>
        %alloc_83 = memref.alloc() : memref<288x1xf32>
        memref.copy %arg19, %alloc_83 : memref<288x1xf32> to memref<288x1xf32>
        %subview_84 = memref.subview %alloc_83[%arg18, 0] [48, 1] [1, 1] : memref<288x1xf32> to memref<48x1xf32, strided<[1, 1], offset: ?>>
        memref.copy %alloc_16, %subview_84 : memref<48x1xf32> to memref<48x1xf32, strided<[1, 1], offset: ?>>
        scf.yield %alloc_83 : memref<288x1xf32>
      }
      %reshape_59 = memref.reshape %10(%8) : (memref<288x1xf32>, memref<1xi64>) -> memref<288xf32>
      %11 = arith.index_cast %arg1 : index to i64
      %12 = arith.uitofp %11 : i64 to f32
      memref.copy %reshape_57, %alloc_17 : memref<288xf32> to memref<288xf32>
      memref.copy %reshape_58, %alloc_18 : memref<288xf32> to memref<288xf32>
      %13:2 = scf.for %arg18 = %c0 to %c288 step %c2 iter_args(%arg19 = %alloc_17, %arg20 = %alloc_18) -> (memref<288xf32>, memref<288xf32>) {
        %27 = arith.remui %arg18, %c48 : index
        %28 = arith.index_cast %27 : index to i64
        %29 = arith.uitofp %28 : i64 to f32
        %30 = arith.divf %29, %cst_1 : f32
        %31 = llvm.intr.pow(%cst_2, %30)  : (f32, f32) -> f32
        %32 = arith.divf %cst_0, %31 : f32
        %33 = arith.mulf %12, %32 : f32
        %34 = llvm.intr.cos(%33)  : (f32) -> f32
        %35 = llvm.intr.sin(%33)  : (f32) -> f32
        %alloc_82 = memref.alloc() : memref<288xf32>
        memref.copy %arg19, %alloc_82 : memref<288xf32> to memref<288xf32>
        %36 = func.call @rot(%alloc_82, %arg18, %34, %35) : (memref<288xf32>, index, f32, f32) -> memref<288xf32>
        %37 = arith.cmpi ult, %arg18, %c288 : index
        %alloc_83 = memref.alloc() : memref<288xf32>
        %38 = scf.if %37 -> (memref<288xf32>) {
          memref.copy %arg20, %alloc_83 : memref<288xf32> to memref<288xf32>
          %39 = func.call @rot(%alloc_83, %arg18, %34, %35) : (memref<288xf32>, index, f32, f32) -> memref<288xf32>
          scf.yield %39 : memref<288xf32>
        } else {
          scf.yield %arg20 : memref<288xf32>
        }
        scf.yield %36, %38 : memref<288xf32>, memref<288xf32>
      }
      %subview_60 = memref.subview %arg2[%arg16, %arg1, 0] [1, 1, 288] [1, 1, 1] : memref<6x256x288xf32> to memref<288xf32, strided<[1], offset: ?>>
      %subview_61 = memref.subview %arg3[%arg16, %arg1, 0] [1, 1, 288] [1, 1, 1] : memref<6x256x288xf32> to memref<288xf32, strided<[1], offset: ?>>
      memref.copy %reshape_59, %subview_61 : memref<288xf32> to memref<288xf32, strided<[1], offset: ?>>
      memref.copy %13#1, %subview_60 : memref<288xf32> to memref<288xf32, strided<[1], offset: ?>>
      %subview_62 = memref.subview %arg2[%arg16, 0, 0] [1, 256, 288] [1, 1, 1] : memref<6x256x288xf32> to memref<256x288xf32, strided<[288, 1], offset: ?>>
      %subview_63 = memref.subview %arg3[%arg16, 0, 0] [1, 256, 288] [1, 1, 1] : memref<6x256x288xf32> to memref<256x288xf32, strided<[288, 1], offset: ?>>
      memref.copy %13#0, %alloc_19 : memref<288xf32> to memref<288xf32>
      memref.copy %subview_62, %alloc_20 : memref<256x288xf32, strided<[288, 1], offset: ?>> to memref<256x288xf32>
      memref.copy %subview_63, %alloc_21 : memref<256x288xf32, strided<[288, 1], offset: ?>> to memref<256x288xf32>
      %14 = func.call @mha(%alloc_19, %alloc_20, %alloc_21, %arg1) : (memref<288xf32>, memref<256x288xf32>, memref<256x288xf32>, index) -> memref<288xf32>
      %subview_64 = memref.subview %arg9[%arg16, 0, 0] [1, 288, 288] [1, 1, 1] : memref<6x288x288xf32> to memref<288x288xf32, strided<[288, 1], offset: ?>>
      %reshape_65 = memref.reshape %14(%6) : (memref<288xf32>, memref<2xi64>) -> memref<288x1xf32>
      memref.copy %alloc_22, %alloc_23 : memref<288x1xf32> to memref<288x1xf32>
      %15 = scf.for %arg18 = %c0 to %c288 step %c48 iter_args(%arg19 = %alloc_23) -> (memref<288x1xf32>) {
        %27 = memref.get_global @__constant_48x1xf32 : memref<48x1xf32>
        %subview_82 = memref.subview %subview_64[%arg18, 0] [48, 288] [1, 1] : memref<288x288xf32, strided<[288, 1], offset: ?>> to memref<48x288xf32, strided<[288, 1], offset: ?>>
        %28 = upmem.alloc_dpus : !upmem.hierarchy<1x6x8>
        scf.for %arg20 = %c0 to %c288 step %c1 {
          %29 = memref.load %reshape_65[%arg20, %c0] : memref<288x1xf32>
          memref.store %29, %alloc_24[%c0, %arg20] : memref<1x288xf32>
        }
        upmem.scatter %subview_82[0, 2304, #map] onto %28 : memref<48x288xf32, strided<[288, 1], offset: ?>> onto !upmem.hierarchy<1x6x8>
        upmem.scatter %alloc_24[9216, 2304, #map1] onto %28 : memref<1x288xf32> onto !upmem.hierarchy<1x6x8>
        upmem.scatter %27[18432, 8, #map2] onto %28 : memref<48x1xf32> onto !upmem.hierarchy<1x6x8>
        upmem.launch_func  @dpu_kernels::@forward %28 : !upmem.hierarchy<1x6x8> 
        upmem.gather %alloc_25[18432, 8, #map2] from %28 : memref<48x1xf32> from !upmem.hierarchy<1x6x8>
        upmem.free_dpus %28 : !upmem.hierarchy<1x6x8>
        %alloc_83 = memref.alloc() : memref<288x1xf32>
        memref.copy %arg19, %alloc_83 : memref<288x1xf32> to memref<288x1xf32>
        %subview_84 = memref.subview %alloc_83[%arg18, 0] [48, 1] [1, 1] : memref<288x1xf32> to memref<48x1xf32, strided<[1, 1], offset: ?>>
        memref.copy %alloc_25, %subview_84 : memref<48x1xf32> to memref<48x1xf32, strided<[1, 1], offset: ?>>
        scf.yield %alloc_83 : memref<288x1xf32>
      }
      %reshape_66 = memref.reshape %15(%8) : (memref<288x1xf32>, memref<1xi64>) -> memref<288xf32>
      %16 = upmem.alloc_dpus : !upmem.hierarchy<1x6x8>
      %17 = memref.get_global @__constant_2xi64_7 : memref<2xi64>
      %reshape_67 = memref.reshape %arg17(%17) : (memref<288xf32>, memref<2xi64>) -> memref<48x6xf32>
      upmem.scatter %reshape_67[0, 48, #map] onto %16 : memref<48x6xf32> onto !upmem.hierarchy<1x6x8>
      %reshape_68 = memref.reshape %reshape_66(%17) : (memref<288xf32>, memref<2xi64>) -> memref<48x6xf32>
      upmem.scatter %reshape_68[192, 48, #map] onto %16 : memref<48x6xf32> onto !upmem.hierarchy<1x6x8>
      %18 = memref.get_global @__constant_48x6xf32 : memref<48x6xf32>
      upmem.scatter %18[384, 48, #map] onto %16 : memref<48x6xf32> onto !upmem.hierarchy<1x6x8>
      upmem.launch_func  @dpu_kernels::@forward_3 %16 : !upmem.hierarchy<1x6x8> 
      upmem.gather %alloc_26[384, 48, #map] from %16 : memref<48x6xf32> from !upmem.hierarchy<1x6x8>
      %reshape_69 = memref.reshape %alloc_26(%8) : (memref<48x6xf32>, memref<1xi64>) -> memref<288xf32>
      upmem.free_dpus %16 : !upmem.hierarchy<1x6x8>
      %subview_70 = memref.subview %arg13[%arg16, 0] [1, 288] [1, 1] : memref<6x288xf32> to memref<288xf32, strided<[1], offset: ?>>
      memref.copy %reshape_69, %alloc_27 : memref<288xf32> to memref<288xf32>
      memref.copy %subview_70, %alloc_28 : memref<288xf32, strided<[1], offset: ?>> to memref<288xf32>
      %19 = func.call @rmsnorm(%alloc_27, %alloc_28) : (memref<288xf32>, memref<288xf32>) -> memref<288xf32>
      %subview_71 = memref.subview %arg10[%arg16, 0, 0] [1, 768, 288] [1, 1, 1] : memref<6x768x288xf32> to memref<768x288xf32, strided<[288, 1], offset: ?>>
      %subview_72 = memref.subview %arg12[%arg16, 0, 0] [1, 768, 288] [1, 1, 1] : memref<6x768x288xf32> to memref<768x288xf32, strided<[288, 1], offset: ?>>
      %reshape_73 = memref.reshape %19(%6) : (memref<288xf32>, memref<2xi64>) -> memref<288x1xf32>
      memref.copy %alloc_29, %alloc_30 : memref<768x1xf32> to memref<768x1xf32>
      %20 = scf.for %arg18 = %c0 to %c768 step %c48 iter_args(%arg19 = %alloc_30) -> (memref<768x1xf32>) {
        %27 = memref.get_global @__constant_48x1xf32 : memref<48x1xf32>
        %subview_82 = memref.subview %subview_71[%arg18, 0] [48, 288] [1, 1] : memref<768x288xf32, strided<[288, 1], offset: ?>> to memref<48x288xf32, strided<[288, 1], offset: ?>>
        %28 = upmem.alloc_dpus : !upmem.hierarchy<1x6x8>
        scf.for %arg20 = %c0 to %c288 step %c1 {
          %29 = memref.load %reshape_73[%arg20, %c0] : memref<288x1xf32>
          memref.store %29, %alloc_31[%c0, %arg20] : memref<1x288xf32>
        }
        upmem.scatter %subview_82[0, 2304, #map] onto %28 : memref<48x288xf32, strided<[288, 1], offset: ?>> onto !upmem.hierarchy<1x6x8>
        upmem.scatter %alloc_31[9216, 2304, #map1] onto %28 : memref<1x288xf32> onto !upmem.hierarchy<1x6x8>
        upmem.scatter %27[18432, 8, #map2] onto %28 : memref<48x1xf32> onto !upmem.hierarchy<1x6x8>
        upmem.launch_func  @dpu_kernels::@forward %28 : !upmem.hierarchy<1x6x8> 
        upmem.gather %alloc_32[18432, 8, #map2] from %28 : memref<48x1xf32> from !upmem.hierarchy<1x6x8>
        upmem.free_dpus %28 : !upmem.hierarchy<1x6x8>
        %alloc_83 = memref.alloc() : memref<768x1xf32>
        memref.copy %arg19, %alloc_83 : memref<768x1xf32> to memref<768x1xf32>
        %subview_84 = memref.subview %alloc_83[%arg18, 0] [48, 1] [1, 1] : memref<768x1xf32> to memref<48x1xf32, strided<[1, 1], offset: ?>>
        memref.copy %alloc_32, %subview_84 : memref<48x1xf32> to memref<48x1xf32, strided<[1, 1], offset: ?>>
        scf.yield %alloc_83 : memref<768x1xf32>
      }
      %21 = memref.get_global @__constant_1xi64_6 : memref<1xi64>
      %reshape_74 = memref.reshape %20(%21) : (memref<768x1xf32>, memref<1xi64>) -> memref<768xf32>
      memref.copy %alloc_33, %alloc_34 : memref<768x1xf32> to memref<768x1xf32>
      %22 = scf.for %arg18 = %c0 to %c768 step %c48 iter_args(%arg19 = %alloc_34) -> (memref<768x1xf32>) {
        %27 = memref.get_global @__constant_48x1xf32 : memref<48x1xf32>
        %subview_82 = memref.subview %subview_72[%arg18, 0] [48, 288] [1, 1] : memref<768x288xf32, strided<[288, 1], offset: ?>> to memref<48x288xf32, strided<[288, 1], offset: ?>>
        %28 = upmem.alloc_dpus : !upmem.hierarchy<1x6x8>
        scf.for %arg20 = %c0 to %c288 step %c1 {
          %29 = memref.load %reshape_73[%arg20, %c0] : memref<288x1xf32>
          memref.store %29, %alloc_35[%c0, %arg20] : memref<1x288xf32>
        }
        upmem.scatter %subview_82[0, 2304, #map] onto %28 : memref<48x288xf32, strided<[288, 1], offset: ?>> onto !upmem.hierarchy<1x6x8>
        upmem.scatter %alloc_35[9216, 2304, #map1] onto %28 : memref<1x288xf32> onto !upmem.hierarchy<1x6x8>
        upmem.scatter %27[18432, 8, #map2] onto %28 : memref<48x1xf32> onto !upmem.hierarchy<1x6x8>
        upmem.launch_func  @dpu_kernels::@forward %28 : !upmem.hierarchy<1x6x8> 
        upmem.gather %alloc_36[18432, 8, #map2] from %28 : memref<48x1xf32> from !upmem.hierarchy<1x6x8>
        upmem.free_dpus %28 : !upmem.hierarchy<1x6x8>
        %alloc_83 = memref.alloc() : memref<768x1xf32>
        memref.copy %arg19, %alloc_83 : memref<768x1xf32> to memref<768x1xf32>
        %subview_84 = memref.subview %alloc_83[%arg18, 0] [48, 1] [1, 1] : memref<768x1xf32> to memref<48x1xf32, strided<[1, 1], offset: ?>>
        memref.copy %alloc_36, %subview_84 : memref<48x1xf32> to memref<48x1xf32, strided<[1, 1], offset: ?>>
        scf.yield %alloc_83 : memref<768x1xf32>
      }
      %reshape_75 = memref.reshape %22(%21) : (memref<768x1xf32>, memref<1xi64>) -> memref<768xf32>
      memref.copy %reshape_74, %alloc_37 : memref<768xf32> to memref<768xf32>
      %23 = scf.for %arg18 = %c0 to %c768 step %c1 iter_args(%arg19 = %alloc_37) -> (memref<768xf32>) {
        %27 = memref.load %arg19[%arg18] : memref<768xf32>
        %28 = memref.load %reshape_75[%arg18] : memref<768xf32>
        %29 = llvm.intr.exp(%27)  : (f32) -> f32
        %30 = arith.addf %29, %cst_0 : f32
        %31 = arith.divf %cst_0, %30 : f32
        %32 = arith.mulf %28, %31 : f32
        %alloc_82 = memref.alloc() : memref<768xf32>
        memref.copy %arg19, %alloc_82 : memref<768xf32> to memref<768xf32>
        memref.store %32, %alloc_82[%arg18] : memref<768xf32>
        scf.yield %alloc_82 : memref<768xf32>
      }
      %subview_76 = memref.subview %arg11[%arg16, 0, 0] [1, 288, 768] [1, 1, 1] : memref<6x288x768xf32> to memref<288x768xf32, strided<[768, 1], offset: ?>>
      %24 = memref.get_global @__constant_2xi64_8 : memref<2xi64>
      %reshape_77 = memref.reshape %23(%24) : (memref<768xf32>, memref<2xi64>) -> memref<768x1xf32>
      memref.copy %alloc_38, %alloc_39 : memref<288x1xf32> to memref<288x1xf32>
      %25 = scf.for %arg18 = %c0 to %c288 step %c48 iter_args(%arg19 = %alloc_39) -> (memref<288x1xf32>) {
        %27 = memref.get_global @__constant_48x1xf32 : memref<48x1xf32>
        %subview_82 = memref.subview %subview_76[%arg18, 0] [48, 768] [1, 1] : memref<288x768xf32, strided<[768, 1], offset: ?>> to memref<48x768xf32, strided<[768, 1], offset: ?>>
        %28 = upmem.alloc_dpus : !upmem.hierarchy<1x6x8>
        scf.for %arg20 = %c0 to %c768 step %c1 {
          %29 = memref.load %reshape_77[%arg20, %c0] : memref<768x1xf32>
          memref.store %29, %alloc_40[%c0, %arg20] : memref<1x768xf32>
        }
        upmem.scatter %subview_82[0, 6144, #map] onto %28 : memref<48x768xf32, strided<[768, 1], offset: ?>> onto !upmem.hierarchy<1x6x8>
        upmem.scatter %alloc_40[24576, 6144, #map1] onto %28 : memref<1x768xf32> onto !upmem.hierarchy<1x6x8>
        upmem.scatter %27[49152, 8, #map2] onto %28 : memref<48x1xf32> onto !upmem.hierarchy<1x6x8>
        upmem.launch_func  @dpu_kernels::@forward_6 %28 : !upmem.hierarchy<1x6x8> 
        upmem.gather %alloc_41[49152, 8, #map2] from %28 : memref<48x1xf32> from !upmem.hierarchy<1x6x8>
        upmem.free_dpus %28 : !upmem.hierarchy<1x6x8>
        %alloc_83 = memref.alloc() : memref<288x1xf32>
        memref.copy %arg19, %alloc_83 : memref<288x1xf32> to memref<288x1xf32>
        %subview_84 = memref.subview %alloc_83[%arg18, 0] [48, 1] [1, 1] : memref<288x1xf32> to memref<48x1xf32, strided<[1, 1], offset: ?>>
        memref.copy %alloc_41, %subview_84 : memref<48x1xf32> to memref<48x1xf32, strided<[1, 1], offset: ?>>
        scf.yield %alloc_83 : memref<288x1xf32>
      }
      %reshape_78 = memref.reshape %25(%8) : (memref<288x1xf32>, memref<1xi64>) -> memref<288xf32>
      %26 = upmem.alloc_dpus : !upmem.hierarchy<1x6x8>
      upmem.scatter %reshape_67[0, 48, #map] onto %26 : memref<48x6xf32> onto !upmem.hierarchy<1x6x8>
      %reshape_79 = memref.reshape %reshape_78(%17) : (memref<288xf32>, memref<2xi64>) -> memref<48x6xf32>
      upmem.scatter %reshape_79[192, 48, #map] onto %26 : memref<48x6xf32> onto !upmem.hierarchy<1x6x8>
      upmem.scatter %18[384, 48, #map] onto %26 : memref<48x6xf32> onto !upmem.hierarchy<1x6x8>
      upmem.launch_func  @dpu_kernels::@forward_3 %26 : !upmem.hierarchy<1x6x8> 
      %alloc_80 = memref.alloc() : memref<48x6xf32>
      upmem.gather %alloc_80[384, 48, #map] from %26 : memref<48x6xf32> from !upmem.hierarchy<1x6x8>
      %reshape_81 = memref.reshape %alloc_80(%8) : (memref<48x6xf32>, memref<1xi64>) -> memref<288xf32>
      upmem.free_dpus %26 : !upmem.hierarchy<1x6x8>
      scf.yield %reshape_81 : memref<288xf32>
    }
    %alloc_42 = memref.alloc() : memref<288xf32>
    memref.copy %0, %alloc_42 : memref<288xf32> to memref<288xf32>
    %alloc_43 = memref.alloc() : memref<288xf32>
    memref.copy %arg14, %alloc_43 : memref<288xf32> to memref<288xf32>
    %1 = call @rmsnorm(%alloc_42, %alloc_43) : (memref<288xf32>, memref<288xf32>) -> memref<288xf32>
    %alloc_44 = memref.alloc() : memref<32768x288xf32>
    scf.for %arg16 = %c0 to %c32768 step %c1 {
      scf.for %arg17 = %c0 to %c288 step %c1 {
        memref.store %cst, %alloc_44[%arg16, %arg17] : memref<32768x288xf32>
      }
    }
    %subview_45 = memref.subview %alloc_44[0, 0] [32000, 288] [1, 1] : memref<32768x288xf32> to memref<32000x288xf32, strided<[288, 1]>>
    memref.copy %arg15, %subview_45 : memref<32000x288xf32> to memref<32000x288xf32, strided<[288, 1]>>
    %2 = memref.get_global @__constant_2xi64_6 : memref<2xi64>
    %reshape = memref.reshape %1(%2) : (memref<288xf32>, memref<2xi64>) -> memref<288x1xf32>
    %alloc_46 = memref.alloc() : memref<32768x1xf32>
    %alloc_47 = memref.alloc() : memref<32768x1xf32>
    memref.copy %alloc_46, %alloc_47 : memref<32768x1xf32> to memref<32768x1xf32>
    %alloc_48 = memref.alloc() : memref<1x288xf32>
    %alloc_49 = memref.alloc() : memref<256x1xf32>
    %3 = scf.for %arg16 = %c0 to %c32768 step %c256 iter_args(%arg17 = %alloc_47) -> (memref<32768x1xf32>) {
      %5 = memref.get_global @__constant_256x1xf32 : memref<256x1xf32>
      %subview_52 = memref.subview %alloc_44[%arg16, 0] [256, 288] [1, 1] : memref<32768x288xf32> to memref<256x288xf32, strided<[288, 1], offset: ?>>
      %6 = upmem.alloc_dpus : !upmem.hierarchy<2x8x16>
      scf.for %arg18 = %c0 to %c288 step %c1 {
        %7 = memref.load %reshape[%arg18, %c0] : memref<288x1xf32>
        memref.store %7, %alloc_48[%c0, %arg18] : memref<1x288xf32>
      }
      upmem.scatter %subview_52[0, 4608, #map3] onto %6 : memref<256x288xf32, strided<[288, 1], offset: ?>> onto !upmem.hierarchy<2x8x16>
      upmem.scatter %alloc_48[18432, 4608, #map1] onto %6 : memref<1x288xf32> onto !upmem.hierarchy<2x8x16>
      upmem.scatter %5[36864, 16, #map4] onto %6 : memref<256x1xf32> onto !upmem.hierarchy<2x8x16>
      upmem.launch_func  @dpu_kernels::@forward_8 %6 : !upmem.hierarchy<2x8x16> 
      upmem.gather %alloc_49[36864, 16, #map4] from %6 : memref<256x1xf32> from !upmem.hierarchy<2x8x16>
      upmem.free_dpus %6 : !upmem.hierarchy<2x8x16>
      %alloc_53 = memref.alloc() : memref<32768x1xf32>
      memref.copy %arg17, %alloc_53 : memref<32768x1xf32> to memref<32768x1xf32>
      %subview_54 = memref.subview %alloc_53[%arg16, 0] [256, 1] [1, 1] : memref<32768x1xf32> to memref<256x1xf32, strided<[1, 1], offset: ?>>
      memref.copy %alloc_49, %subview_54 : memref<256x1xf32> to memref<256x1xf32, strided<[1, 1], offset: ?>>
      scf.yield %alloc_53 : memref<32768x1xf32>
    }
    %4 = memref.get_global @__constant_1xi64_7 : memref<1xi64>
    %reshape_50 = memref.reshape %3(%4) : (memref<32768x1xf32>, memref<1xi64>) -> memref<32768xf32>
    %subview_51 = memref.subview %reshape_50[0] [32000] [1] : memref<32768xf32> to memref<32000xf32, strided<[1]>>
    %cast = memref.cast %subview_51 : memref<32000xf32, strided<[1]>> to memref<32000xf32>
    return %cast : memref<32000xf32>
  }
  func.func @rot(%arg0: memref<288xf32>, %arg1: index, %arg2: f32, %arg3: f32) -> memref<288xf32> {
    %c1 = arith.constant 1 : index
    %0 = arith.addi %arg1, %c1 : index
    %1 = memref.load %arg0[%arg1] : memref<288xf32>
    %2 = memref.load %arg0[%0] : memref<288xf32>
    %3 = arith.mulf %1, %arg2 : f32
    %4 = arith.mulf %2, %arg3 : f32
    %5 = arith.subf %3, %4 : f32
    %alloc = memref.alloc() : memref<288xf32>
    memref.copy %arg0, %alloc : memref<288xf32> to memref<288xf32>
    memref.store %5, %alloc[%arg1] : memref<288xf32>
    %alloc_0 = memref.alloc() : memref<288xf32>
    memref.copy %alloc, %alloc_0 : memref<288xf32> to memref<288xf32>
    memref.store %5, %alloc_0[%arg1] : memref<288xf32>
    return %alloc_0 : memref<288xf32>
  }
  func.func @mha(%arg0: memref<288xf32>, %arg1: memref<256x288xf32>, %arg2: memref<256x288xf32>, %arg3: index) -> memref<288xf32> {
    %c256 = arith.constant 256 : index
    %c7 = arith.constant 7 : index
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    %c48 = arith.constant 48 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 6.92820311 : f32
    %cst_1 = arith.constant 0xFF800000 : f32
    %0 = arith.addi %arg3, %c1 : index
    %alloc = memref.alloc() : memref<256xf32>
    scf.for %arg4 = %c0 to %c256 step %c1 {
      memref.store %cst_1, %alloc[%arg4] : memref<256xf32>
    }
    %alloc_2 = memref.alloc() : memref<288xf32>
    %alloc_3 = memref.alloc() : memref<288xf32>
    memref.copy %alloc_2, %alloc_3 : memref<288xf32> to memref<288xf32>
    %alloc_4 = memref.alloc() : memref<256xf32>
    %alloc_5 = memref.alloc() : memref<48xf32>
    %alloc_6 = memref.alloc() : memref<48xf32>
    %alloc_7 = memref.alloc() : memref<8x6xf32>
    %alloc_8 = memref.alloc() : memref<1xf32>
    %alloc_9 = memref.alloc() : memref<f32>
    %alloc_10 = memref.alloc() : memref<f32>
    %alloc_11 = memref.alloc() : memref<256xf32>
    %alloc_12 = memref.alloc() : memref<48xf32>
    %alloc_13 = memref.alloc() : memref<48xf32>
    %alloc_14 = memref.alloc() : memref<48xf32>
    %alloc_15 = memref.alloc() : memref<8xf32>
    %alloc_16 = memref.alloc() : memref<8x6xf32>
    %1 = scf.for %arg4 = %c0 to %c6 step %c1 iter_args(%arg5 = %alloc_3) -> (memref<288xf32>) {
      %2 = arith.muli %arg4, %c48 : index
      memref.copy %alloc, %alloc_4 : memref<256xf32> to memref<256xf32>
      %3 = scf.for %arg6 = %c0 to %0 step %c1 iter_args(%arg7 = %alloc_4) -> (memref<256xf32>) {
        %subview_18 = memref.subview %arg0[%2] [48] [1] : memref<288xf32> to memref<48xf32, strided<[1], offset: ?>>
        %subview_19 = memref.subview %arg1[%arg6, %2] [1, 48] [1, 1] : memref<256x288xf32> to memref<48xf32, strided<[1], offset: ?>>
        %6 = upmem.alloc_dpus : !upmem.hierarchy<1x1x8>
        %7 = memref.get_global @__constant_2xi64_3 : memref<2xi64>
        memref.copy %subview_18, %alloc_5 : memref<48xf32, strided<[1], offset: ?>> to memref<48xf32>
        %reshape = memref.reshape %alloc_5(%7) : (memref<48xf32>, memref<2xi64>) -> memref<8x6xf32>
        upmem.scatter %reshape[0, 48, #map5] onto %6 : memref<8x6xf32> onto !upmem.hierarchy<1x1x8>
        memref.copy %subview_19, %alloc_6 : memref<48xf32, strided<[1], offset: ?>> to memref<48xf32>
        %reshape_20 = memref.reshape %alloc_6(%7) : (memref<48xf32>, memref<2xi64>) -> memref<8x6xf32>
        upmem.scatter %reshape_20[192, 48, #map5] onto %6 : memref<8x6xf32> onto !upmem.hierarchy<1x1x8>
        %8 = memref.get_global @__constant_8x6xf32 : memref<8x6xf32>
        upmem.scatter %8[384, 48, #map5] onto %6 : memref<8x6xf32> onto !upmem.hierarchy<1x1x8>
        upmem.launch_func  @dpu_kernels::@mha %6 : !upmem.hierarchy<1x1x8> 
        upmem.gather %alloc_7[384, 48, #map5] from %6 : memref<8x6xf32> from !upmem.hierarchy<1x1x8>
        %9 = memref.get_global @__constant_1xi64_3 : memref<1xi64>
        %reshape_21 = memref.reshape %alloc_7(%9) : (memref<8x6xf32>, memref<1xi64>) -> memref<48xf32>
        upmem.free_dpus %6 : !upmem.hierarchy<1x1x8>
        %10 = memref.get_global @__constant_xf32_0 : memref<f32>
        memref.copy %10, %alloc_9 : memref<f32> to memref<f32>
        scf.for %arg8 = %c0 to %c48 step %c1 {
          %17 = memref.load %reshape_21[%arg8] : memref<48xf32>
          %18 = memref.load %alloc_9[] : memref<f32>
          %19 = arith.addf %17, %18 : f32
          memref.store %19, %alloc_9[] : memref<f32>
        }
        %11 = memref.load %alloc_9[] : memref<f32>
        memref.store %11, %alloc_8[%c0] : memref<1xf32>
        memref.copy %10, %alloc_10 : memref<f32> to memref<f32>
        %12 = memref.load %alloc_8[%c0] : memref<1xf32>
        %13 = memref.load %alloc_10[] : memref<f32>
        %14 = arith.addf %12, %13 : f32
        memref.store %14, %alloc_10[] : memref<f32>
        %15 = memref.load %alloc_10[] : memref<f32>
        %16 = arith.divf %15, %cst_0 : f32
        %alloc_22 = memref.alloc() : memref<256xf32>
        memref.copy %arg7, %alloc_22 : memref<256xf32> to memref<256xf32>
        memref.store %16, %alloc_22[%arg6] : memref<256xf32>
        scf.yield %alloc_22 : memref<256xf32>
      }
      memref.copy %3, %alloc_11 : memref<256xf32> to memref<256xf32>
      %4 = func.call @softmax(%alloc_11) : (memref<256xf32>) -> memref<256xf32>
      scf.for %arg6 = %c0 to %c48 step %c1 {
        memref.store %cst, %alloc_12[%arg6] : memref<48xf32>
      }
      memref.copy %alloc_12, %alloc_13 : memref<48xf32> to memref<48xf32>
      %5 = scf.for %arg6 = %c0 to %0 step %c1 iter_args(%arg7 = %alloc_13) -> (memref<48xf32>) {
        %subview_18 = memref.subview %arg2[%arg6, %2] [1, 48] [1, 1] : memref<256x288xf32> to memref<48xf32, strided<[1], offset: ?>>
        %6 = memref.load %4[%arg6] : memref<256xf32>
        %7 = upmem.alloc_dpus : !upmem.hierarchy<1x1x8>
        %8 = memref.get_global @__constant_2xi64_3 : memref<2xi64>
        memref.copy %subview_18, %alloc_14 : memref<48xf32, strided<[1], offset: ?>> to memref<48xf32>
        %reshape = memref.reshape %alloc_14(%8) : (memref<48xf32>, memref<2xi64>) -> memref<8x6xf32>
        upmem.scatter %reshape[0, 48, #map5] onto %7 : memref<8x6xf32> onto !upmem.hierarchy<1x1x8>
        memref.store %6, %alloc_15[%c0] : memref<8xf32>
        memref.store %6, %alloc_15[%c1] : memref<8xf32>
        memref.store %6, %alloc_15[%c2] : memref<8xf32>
        memref.store %6, %alloc_15[%c3] : memref<8xf32>
        memref.store %6, %alloc_15[%c4] : memref<8xf32>
        memref.store %6, %alloc_15[%c5] : memref<8xf32>
        memref.store %6, %alloc_15[%c6] : memref<8xf32>
        memref.store %6, %alloc_15[%c7] : memref<8xf32>
        upmem.scatter %alloc_15[192, 8, #map1] onto %7 : memref<8xf32> onto !upmem.hierarchy<1x1x8>
        %9 = memref.get_global @__constant_8x6xf32 : memref<8x6xf32>
        upmem.scatter %9[224, 48, #map5] onto %7 : memref<8x6xf32> onto !upmem.hierarchy<1x1x8>
        upmem.launch_func  @dpu_kernels::@mha_9 %7 : !upmem.hierarchy<1x1x8> 
        upmem.gather %alloc_16[224, 48, #map5] from %7 : memref<8x6xf32> from !upmem.hierarchy<1x1x8>
        %10 = memref.get_global @__constant_1xi64_3 : memref<1xi64>
        %reshape_19 = memref.reshape %alloc_16(%10) : (memref<8x6xf32>, memref<1xi64>) -> memref<48xf32>
        upmem.free_dpus %7 : !upmem.hierarchy<1x1x8>
        %11 = upmem.alloc_dpus : !upmem.hierarchy<1x1x8>
        %reshape_20 = memref.reshape %arg7(%8) : (memref<48xf32>, memref<2xi64>) -> memref<8x6xf32>
        upmem.scatter %reshape_20[0, 48, #map5] onto %11 : memref<8x6xf32> onto !upmem.hierarchy<1x1x8>
        %reshape_21 = memref.reshape %reshape_19(%8) : (memref<48xf32>, memref<2xi64>) -> memref<8x6xf32>
        upmem.scatter %reshape_21[192, 48, #map5] onto %11 : memref<8x6xf32> onto !upmem.hierarchy<1x1x8>
        upmem.scatter %9[384, 48, #map5] onto %11 : memref<8x6xf32> onto !upmem.hierarchy<1x1x8>
        upmem.launch_func  @dpu_kernels::@forward_3 %11 : !upmem.hierarchy<1x1x8> 
        %alloc_22 = memref.alloc() : memref<8x6xf32>
        upmem.gather %alloc_22[384, 48, #map5] from %11 : memref<8x6xf32> from !upmem.hierarchy<1x1x8>
        %reshape_23 = memref.reshape %alloc_22(%10) : (memref<8x6xf32>, memref<1xi64>) -> memref<48xf32>
        upmem.free_dpus %11 : !upmem.hierarchy<1x1x8>
        scf.yield %reshape_23 : memref<48xf32>
      }
      %alloc_17 = memref.alloc() : memref<288xf32>
      memref.copy %arg5, %alloc_17 : memref<288xf32> to memref<288xf32>
      %subview = memref.subview %alloc_17[%2] [48] [1] : memref<288xf32> to memref<48xf32, strided<[1], offset: ?>>
      memref.copy %5, %subview : memref<48xf32> to memref<48xf32, strided<[1], offset: ?>>
      scf.yield %alloc_17 : memref<288xf32>
    }
    return %1 : memref<288xf32>
  }
  func.func @rmsnorm(%arg0: memref<288xf32>, %arg1: memref<288xf32>) -> memref<288xf32> {
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
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c288 = arith.constant 288 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 9.99999974E-6 : f32
    %cst_0 = arith.constant 2.880000e+02 : f32
    %0 = upmem.alloc_dpus : !upmem.hierarchy<1x1x16>
    %1 = memref.get_global @__constant_2xi64_1 : memref<2xi64>
    %reshape = memref.reshape %arg0(%1) : (memref<288xf32>, memref<2xi64>) -> memref<16x18xf32>
    upmem.scatter %reshape[0, 288, #map5] onto %0 : memref<16x18xf32> onto !upmem.hierarchy<1x1x16>
    upmem.scatter %reshape[1152, 288, #map5] onto %0 : memref<16x18xf32> onto !upmem.hierarchy<1x1x16>
    %2 = memref.get_global @__constant_16x18xf32 : memref<16x18xf32>
    upmem.scatter %2[2304, 288, #map5] onto %0 : memref<16x18xf32> onto !upmem.hierarchy<1x1x16>
    upmem.launch_func  @dpu_kernels::@rmsnorm %0 : !upmem.hierarchy<1x1x16> 
    %alloc = memref.alloc() : memref<16x18xf32>
    upmem.gather %alloc[2304, 288, #map5] from %0 : memref<16x18xf32> from !upmem.hierarchy<1x1x16>
    %3 = memref.get_global @__constant_1xi64_1 : memref<1xi64>
    %reshape_1 = memref.reshape %alloc(%3) : (memref<16x18xf32>, memref<1xi64>) -> memref<288xf32>
    upmem.free_dpus %0 : !upmem.hierarchy<1x1x16>
    %4 = memref.get_global @__constant_xf32_0 : memref<f32>
    %alloc_2 = memref.alloc() : memref<1xf32>
    %alloc_3 = memref.alloc() : memref<f32>
    memref.copy %4, %alloc_3 : memref<f32> to memref<f32>
    scf.for %arg2 = %c0 to %c288 step %c1 {
      %17 = memref.load %reshape_1[%arg2] : memref<288xf32>
      %18 = memref.load %alloc_3[] : memref<f32>
      %19 = arith.addf %17, %18 : f32
      memref.store %19, %alloc_3[] : memref<f32>
    }
    %5 = memref.load %alloc_3[] : memref<f32>
    memref.store %5, %alloc_2[%c0] : memref<1xf32>
    %alloc_4 = memref.alloc() : memref<f32>
    memref.copy %4, %alloc_4 : memref<f32> to memref<f32>
    %6 = memref.load %alloc_2[%c0] : memref<1xf32>
    %7 = memref.load %alloc_4[] : memref<f32>
    %8 = arith.addf %6, %7 : f32
    memref.store %8, %alloc_4[] : memref<f32>
    %9 = memref.load %alloc_4[] : memref<f32>
    %10 = arith.divf %9, %cst_0 : f32
    %11 = arith.addf %10, %cst : f32
    %12 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %13 = llvm.intr.sqrt(%11)  : (f32) -> f32
    %14 = llvm.fdiv %12, %13  : f32
    %15 = upmem.alloc_dpus : !upmem.hierarchy<1x1x16>
    upmem.scatter %reshape[0, 288, #map5] onto %15 : memref<16x18xf32> onto !upmem.hierarchy<1x1x16>
    %alloc_5 = memref.alloc() : memref<16xf32>
    memref.store %14, %alloc_5[%c0] : memref<16xf32>
    memref.store %14, %alloc_5[%c1] : memref<16xf32>
    memref.store %14, %alloc_5[%c2] : memref<16xf32>
    memref.store %14, %alloc_5[%c3] : memref<16xf32>
    memref.store %14, %alloc_5[%c4] : memref<16xf32>
    memref.store %14, %alloc_5[%c5] : memref<16xf32>
    memref.store %14, %alloc_5[%c6] : memref<16xf32>
    memref.store %14, %alloc_5[%c7] : memref<16xf32>
    memref.store %14, %alloc_5[%c8] : memref<16xf32>
    memref.store %14, %alloc_5[%c9] : memref<16xf32>
    memref.store %14, %alloc_5[%c10] : memref<16xf32>
    memref.store %14, %alloc_5[%c11] : memref<16xf32>
    memref.store %14, %alloc_5[%c12] : memref<16xf32>
    memref.store %14, %alloc_5[%c13] : memref<16xf32>
    memref.store %14, %alloc_5[%c14] : memref<16xf32>
    memref.store %14, %alloc_5[%c15] : memref<16xf32>
    upmem.scatter %alloc_5[1152, 16, #map1] onto %15 : memref<16xf32> onto !upmem.hierarchy<1x1x16>
    upmem.scatter %2[1216, 288, #map5] onto %15 : memref<16x18xf32> onto !upmem.hierarchy<1x1x16>
    upmem.launch_func  @dpu_kernels::@rmsnorm_11 %15 : !upmem.hierarchy<1x1x16> 
    %alloc_6 = memref.alloc() : memref<16x18xf32>
    upmem.gather %alloc_6[1216, 288, #map5] from %15 : memref<16x18xf32> from !upmem.hierarchy<1x1x16>
    %reshape_7 = memref.reshape %alloc_6(%3) : (memref<16x18xf32>, memref<1xi64>) -> memref<288xf32>
    upmem.free_dpus %15 : !upmem.hierarchy<1x1x16>
    %16 = upmem.alloc_dpus : !upmem.hierarchy<1x1x16>
    %reshape_8 = memref.reshape %reshape_7(%1) : (memref<288xf32>, memref<2xi64>) -> memref<16x18xf32>
    upmem.scatter %reshape_8[0, 288, #map5] onto %16 : memref<16x18xf32> onto !upmem.hierarchy<1x1x16>
    %reshape_9 = memref.reshape %arg1(%1) : (memref<288xf32>, memref<2xi64>) -> memref<16x18xf32>
    upmem.scatter %reshape_9[1152, 288, #map5] onto %16 : memref<16x18xf32> onto !upmem.hierarchy<1x1x16>
    upmem.scatter %2[2304, 288, #map5] onto %16 : memref<16x18xf32> onto !upmem.hierarchy<1x1x16>
    upmem.launch_func  @dpu_kernels::@rmsnorm %16 : !upmem.hierarchy<1x1x16> 
    %alloc_10 = memref.alloc() : memref<16x18xf32>
    upmem.gather %alloc_10[2304, 288, #map5] from %16 : memref<16x18xf32> from !upmem.hierarchy<1x1x16>
    %reshape_11 = memref.reshape %alloc_10(%3) : (memref<16x18xf32>, memref<1xi64>) -> memref<288xf32>
    upmem.free_dpus %16 : !upmem.hierarchy<1x1x16>
    return %reshape_11 : memref<288xf32>
  }
  func.func @softmax(%arg0: memref<256xf32>) -> memref<256xf32> {
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
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_xf32 : memref<f32>
    %alloc = memref.alloc() : memref<1xf32>
    %alloc_0 = memref.alloc() : memref<f32>
    memref.copy %0, %alloc_0 : memref<f32> to memref<f32>
    scf.for %arg1 = %c0 to %c256 step %c1 {
      %17 = memref.load %arg0[%arg1] : memref<256xf32>
      %18 = memref.load %alloc_0[] : memref<f32>
      %19 = arith.maximumf %17, %18 : f32
      memref.store %19, %alloc_0[] : memref<f32>
    }
    %1 = memref.load %alloc_0[] : memref<f32>
    memref.store %1, %alloc[%c0] : memref<1xf32>
    %alloc_1 = memref.alloc() : memref<f32>
    memref.copy %0, %alloc_1 : memref<f32> to memref<f32>
    %2 = memref.load %alloc[%c0] : memref<1xf32>
    %3 = memref.load %alloc_1[] : memref<f32>
    %4 = arith.maximumf %2, %3 : f32
    memref.store %4, %alloc_1[] : memref<f32>
    %5 = memref.load %alloc_1[] : memref<f32>
    %6 = upmem.alloc_dpus : !upmem.hierarchy<1x8x16>
    %7 = memref.get_global @__constant_2xi64_2 : memref<2xi64>
    %reshape = memref.reshape %arg0(%7) : (memref<256xf32>, memref<2xi64>) -> memref<128x2xf32>
    upmem.scatter %reshape[0, 32, #map3] onto %6 : memref<128x2xf32> onto !upmem.hierarchy<1x8x16>
    %alloc_2 = memref.alloc() : memref<16xf32>
    memref.store %5, %alloc_2[%c0] : memref<16xf32>
    memref.store %5, %alloc_2[%c1] : memref<16xf32>
    memref.store %5, %alloc_2[%c2] : memref<16xf32>
    memref.store %5, %alloc_2[%c3] : memref<16xf32>
    memref.store %5, %alloc_2[%c4] : memref<16xf32>
    memref.store %5, %alloc_2[%c5] : memref<16xf32>
    memref.store %5, %alloc_2[%c6] : memref<16xf32>
    memref.store %5, %alloc_2[%c7] : memref<16xf32>
    memref.store %5, %alloc_2[%c8] : memref<16xf32>
    memref.store %5, %alloc_2[%c9] : memref<16xf32>
    memref.store %5, %alloc_2[%c10] : memref<16xf32>
    memref.store %5, %alloc_2[%c11] : memref<16xf32>
    memref.store %5, %alloc_2[%c12] : memref<16xf32>
    memref.store %5, %alloc_2[%c13] : memref<16xf32>
    memref.store %5, %alloc_2[%c14] : memref<16xf32>
    memref.store %5, %alloc_2[%c15] : memref<16xf32>
    upmem.scatter %alloc_2[128, 16, #map1] onto %6 : memref<16xf32> onto !upmem.hierarchy<1x8x16>
    %8 = memref.get_global @__constant_128x2xf32 : memref<128x2xf32>
    upmem.scatter %8[192, 32, #map3] onto %6 : memref<128x2xf32> onto !upmem.hierarchy<1x8x16>
    upmem.launch_func  @dpu_kernels::@softmax %6 : !upmem.hierarchy<1x8x16> 
    %alloc_3 = memref.alloc() : memref<128x2xf32>
    upmem.gather %alloc_3[192, 32, #map3] from %6 : memref<128x2xf32> from !upmem.hierarchy<1x8x16>
    %9 = memref.get_global @__constant_1xi64_2 : memref<1xi64>
    %reshape_4 = memref.reshape %alloc_3(%9) : (memref<128x2xf32>, memref<1xi64>) -> memref<256xf32>
    upmem.free_dpus %6 : !upmem.hierarchy<1x8x16>
    %alloc_5 = memref.alloc() : memref<256xf32>
    scf.for %arg1 = %c0 to %c256 step %c1 {
      %17 = memref.load %reshape_4[%arg1] : memref<256xf32>
      %18 = llvm.intr.exp(%17)  : (f32) -> f32
      memref.store %18, %alloc_5[%arg1] : memref<256xf32>
    }
    %10 = memref.get_global @__constant_xf32_0 : memref<f32>
    %alloc_6 = memref.alloc() : memref<1xf32>
    %alloc_7 = memref.alloc() : memref<f32>
    memref.copy %10, %alloc_7 : memref<f32> to memref<f32>
    scf.for %arg1 = %c0 to %c256 step %c1 {
      %17 = memref.load %alloc_5[%arg1] : memref<256xf32>
      %18 = memref.load %alloc_7[] : memref<f32>
      %19 = arith.addf %17, %18 : f32
      memref.store %19, %alloc_7[] : memref<f32>
    }
    %11 = memref.load %alloc_7[] : memref<f32>
    memref.store %11, %alloc_6[%c0] : memref<1xf32>
    %alloc_8 = memref.alloc() : memref<f32>
    memref.copy %10, %alloc_8 : memref<f32> to memref<f32>
    %12 = memref.load %alloc_6[%c0] : memref<1xf32>
    %13 = memref.load %alloc_8[] : memref<f32>
    %14 = arith.addf %12, %13 : f32
    memref.store %14, %alloc_8[] : memref<f32>
    %15 = memref.load %alloc_8[] : memref<f32>
    %16 = upmem.alloc_dpus : !upmem.hierarchy<1x8x16>
    %reshape_9 = memref.reshape %alloc_5(%7) : (memref<256xf32>, memref<2xi64>) -> memref<128x2xf32>
    upmem.scatter %reshape_9[0, 32, #map3] onto %16 : memref<128x2xf32> onto !upmem.hierarchy<1x8x16>
    %alloc_10 = memref.alloc() : memref<16xf32>
    memref.store %15, %alloc_10[%c0] : memref<16xf32>
    memref.store %15, %alloc_10[%c1] : memref<16xf32>
    memref.store %15, %alloc_10[%c2] : memref<16xf32>
    memref.store %15, %alloc_10[%c3] : memref<16xf32>
    memref.store %15, %alloc_10[%c4] : memref<16xf32>
    memref.store %15, %alloc_10[%c5] : memref<16xf32>
    memref.store %15, %alloc_10[%c6] : memref<16xf32>
    memref.store %15, %alloc_10[%c7] : memref<16xf32>
    memref.store %15, %alloc_10[%c8] : memref<16xf32>
    memref.store %15, %alloc_10[%c9] : memref<16xf32>
    memref.store %15, %alloc_10[%c10] : memref<16xf32>
    memref.store %15, %alloc_10[%c11] : memref<16xf32>
    memref.store %15, %alloc_10[%c12] : memref<16xf32>
    memref.store %15, %alloc_10[%c13] : memref<16xf32>
    memref.store %15, %alloc_10[%c14] : memref<16xf32>
    memref.store %15, %alloc_10[%c15] : memref<16xf32>
    upmem.scatter %alloc_10[128, 16, #map1] onto %16 : memref<16xf32> onto !upmem.hierarchy<1x8x16>
    upmem.scatter %8[192, 32, #map3] onto %16 : memref<128x2xf32> onto !upmem.hierarchy<1x8x16>
    upmem.launch_func  @dpu_kernels::@softmax_13 %16 : !upmem.hierarchy<1x8x16> 
    %alloc_11 = memref.alloc() : memref<128x2xf32>
    upmem.gather %alloc_11[192, 32, #map3] from %16 : memref<128x2xf32> from !upmem.hierarchy<1x8x16>
    %reshape_12 = memref.reshape %alloc_11(%9) : (memref<128x2xf32>, memref<1xi64>) -> memref<256xf32>
    upmem.free_dpus %16 : !upmem.hierarchy<1x8x16>
    return %reshape_12 : memref<256xf32>
  }
  func.func @rmsnorm_1048576(%arg0: memref<1048576xf32>, %arg1: memref<1048576xf32>) -> memref<1048576xf32> {
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
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 9.99999974E-6 : f32
    %cst_0 = arith.constant 0x49800000 : f32
    %0 = upmem.alloc_dpus : !upmem.hierarchy<4x64x16>
    %1 = memref.get_global @__constant_2xi64 : memref<2xi64>
    %reshape = memref.reshape %arg0(%1) : (memref<1048576xf32>, memref<2xi64>) -> memref<4096x256xf32>
    upmem.scatter %reshape[0, 4096, #map6] onto %0 : memref<4096x256xf32> onto !upmem.hierarchy<4x64x16>
    upmem.scatter %reshape[16384, 4096, #map6] onto %0 : memref<4096x256xf32> onto !upmem.hierarchy<4x64x16>
    %2 = memref.get_global @__constant_4096x256xf32 : memref<4096x256xf32>
    upmem.scatter %2[32768, 4096, #map6] onto %0 : memref<4096x256xf32> onto !upmem.hierarchy<4x64x16>
    upmem.launch_func  @dpu_kernels::@rmsnorm_1048576 %0 : !upmem.hierarchy<4x64x16> 
    %alloc = memref.alloc() : memref<4096x256xf32>
    upmem.gather %alloc[32768, 4096, #map6] from %0 : memref<4096x256xf32> from !upmem.hierarchy<4x64x16>
    %3 = memref.get_global @__constant_1xi64 : memref<1xi64>
    %reshape_1 = memref.reshape %alloc(%3) : (memref<4096x256xf32>, memref<1xi64>) -> memref<1048576xf32>
    upmem.free_dpus %0 : !upmem.hierarchy<4x64x16>
    %4 = memref.get_global @__constant_xf32_0 : memref<f32>
    %alloc_2 = memref.alloc() : memref<1024xf32>
    %alloc_3 = memref.alloc() : memref<f32>
    scf.for %arg2 = %c0 to %c1024 step %c1 {
      %13 = arith.muli %arg2, %c1024 : index
      %subview = memref.subview %reshape_1[%13] [1024] [1] : memref<1048576xf32> to memref<1024xf32, strided<[1], offset: ?>>
      memref.copy %4, %alloc_3 : memref<f32> to memref<f32>
      scf.for %arg3 = %c0 to %c1024 step %c1 {
        %15 = memref.load %subview[%arg3] : memref<1024xf32, strided<[1], offset: ?>>
        %16 = memref.load %alloc_3[] : memref<f32>
        %17 = arith.addf %15, %16 : f32
        memref.store %17, %alloc_3[] : memref<f32>
      }
      %14 = memref.load %alloc_3[] : memref<f32>
      memref.store %14, %alloc_2[%arg2] : memref<1024xf32>
    }
    %alloc_4 = memref.alloc() : memref<f32>
    memref.copy %4, %alloc_4 : memref<f32> to memref<f32>
    scf.for %arg2 = %c0 to %c1024 step %c1 {
      %13 = memref.load %alloc_2[%arg2] : memref<1024xf32>
      %14 = memref.load %alloc_4[] : memref<f32>
      %15 = arith.addf %13, %14 : f32
      memref.store %15, %alloc_4[] : memref<f32>
    }
    %5 = memref.load %alloc_4[] : memref<f32>
    %6 = arith.divf %5, %cst_0 : f32
    %7 = arith.addf %6, %cst : f32
    %8 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %9 = llvm.intr.sqrt(%7)  : (f32) -> f32
    %10 = llvm.fdiv %8, %9  : f32
    %11 = upmem.alloc_dpus : !upmem.hierarchy<4x64x16>
    upmem.scatter %reshape[0, 4096, #map6] onto %11 : memref<4096x256xf32> onto !upmem.hierarchy<4x64x16>
    %alloc_5 = memref.alloc() : memref<16xf32>
    memref.store %10, %alloc_5[%c0] : memref<16xf32>
    memref.store %10, %alloc_5[%c1] : memref<16xf32>
    memref.store %10, %alloc_5[%c2] : memref<16xf32>
    memref.store %10, %alloc_5[%c3] : memref<16xf32>
    memref.store %10, %alloc_5[%c4] : memref<16xf32>
    memref.store %10, %alloc_5[%c5] : memref<16xf32>
    memref.store %10, %alloc_5[%c6] : memref<16xf32>
    memref.store %10, %alloc_5[%c7] : memref<16xf32>
    memref.store %10, %alloc_5[%c8] : memref<16xf32>
    memref.store %10, %alloc_5[%c9] : memref<16xf32>
    memref.store %10, %alloc_5[%c10] : memref<16xf32>
    memref.store %10, %alloc_5[%c11] : memref<16xf32>
    memref.store %10, %alloc_5[%c12] : memref<16xf32>
    memref.store %10, %alloc_5[%c13] : memref<16xf32>
    memref.store %10, %alloc_5[%c14] : memref<16xf32>
    memref.store %10, %alloc_5[%c15] : memref<16xf32>
    upmem.scatter %alloc_5[16384, 16, #map1] onto %11 : memref<16xf32> onto !upmem.hierarchy<4x64x16>
    upmem.scatter %2[16448, 4096, #map6] onto %11 : memref<4096x256xf32> onto !upmem.hierarchy<4x64x16>
    upmem.launch_func  @dpu_kernels::@rmsnorm_1048576_14 %11 : !upmem.hierarchy<4x64x16> 
    %alloc_6 = memref.alloc() : memref<4096x256xf32>
    upmem.gather %alloc_6[16448, 4096, #map6] from %11 : memref<4096x256xf32> from !upmem.hierarchy<4x64x16>
    %reshape_7 = memref.reshape %alloc_6(%3) : (memref<4096x256xf32>, memref<1xi64>) -> memref<1048576xf32>
    upmem.free_dpus %11 : !upmem.hierarchy<4x64x16>
    %12 = upmem.alloc_dpus : !upmem.hierarchy<4x64x16>
    %reshape_8 = memref.reshape %reshape_7(%1) : (memref<1048576xf32>, memref<2xi64>) -> memref<4096x256xf32>
    upmem.scatter %reshape_8[0, 4096, #map6] onto %12 : memref<4096x256xf32> onto !upmem.hierarchy<4x64x16>
    %reshape_9 = memref.reshape %arg1(%1) : (memref<1048576xf32>, memref<2xi64>) -> memref<4096x256xf32>
    upmem.scatter %reshape_9[16384, 4096, #map6] onto %12 : memref<4096x256xf32> onto !upmem.hierarchy<4x64x16>
    upmem.scatter %2[32768, 4096, #map6] onto %12 : memref<4096x256xf32> onto !upmem.hierarchy<4x64x16>
    upmem.launch_func  @dpu_kernels::@rmsnorm_1048576 %12 : !upmem.hierarchy<4x64x16> 
    %alloc_10 = memref.alloc() : memref<4096x256xf32>
    upmem.gather %alloc_10[32768, 4096, #map6] from %12 : memref<4096x256xf32> from !upmem.hierarchy<4x64x16>
    %reshape_11 = memref.reshape %alloc_10(%3) : (memref<4096x256xf32>, memref<1xi64>) -> memref<1048576xf32>
    upmem.free_dpus %12 : !upmem.hierarchy<4x64x16>
    return %reshape_11 : memref<1048576xf32>
  }
  func.func @softmax_1048576(%arg0: memref<1048576xf32>) -> memref<1048576xf32> {
    %c1048576 = arith.constant 1048576 : index
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
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_xf32 : memref<f32>
    %alloc = memref.alloc() : memref<1024xf32>
    %alloc_0 = memref.alloc() : memref<f32>
    scf.for %arg1 = %c0 to %c1024 step %c1 {
      %9 = arith.muli %arg1, %c1024 : index
      %subview = memref.subview %arg0[%9] [1024] [1] : memref<1048576xf32> to memref<1024xf32, strided<[1], offset: ?>>
      memref.copy %0, %alloc_0 : memref<f32> to memref<f32>
      scf.for %arg2 = %c0 to %c1024 step %c1 {
        %11 = memref.load %subview[%arg2] : memref<1024xf32, strided<[1], offset: ?>>
        %12 = memref.load %alloc_0[] : memref<f32>
        %13 = arith.maximumf %11, %12 : f32
        memref.store %13, %alloc_0[] : memref<f32>
      }
      %10 = memref.load %alloc_0[] : memref<f32>
      memref.store %10, %alloc[%arg1] : memref<1024xf32>
    }
    %alloc_1 = memref.alloc() : memref<f32>
    memref.copy %0, %alloc_1 : memref<f32> to memref<f32>
    scf.for %arg1 = %c0 to %c1024 step %c1 {
      %9 = memref.load %alloc[%arg1] : memref<1024xf32>
      %10 = memref.load %alloc_1[] : memref<f32>
      %11 = arith.maximumf %9, %10 : f32
      memref.store %11, %alloc_1[] : memref<f32>
    }
    %1 = memref.load %alloc_1[] : memref<f32>
    %2 = upmem.alloc_dpus : !upmem.hierarchy<4x64x16>
    %3 = memref.get_global @__constant_2xi64 : memref<2xi64>
    %reshape = memref.reshape %arg0(%3) : (memref<1048576xf32>, memref<2xi64>) -> memref<4096x256xf32>
    upmem.scatter %reshape[0, 4096, #map6] onto %2 : memref<4096x256xf32> onto !upmem.hierarchy<4x64x16>
    %alloc_2 = memref.alloc() : memref<16xf32>
    memref.store %1, %alloc_2[%c0] : memref<16xf32>
    memref.store %1, %alloc_2[%c1] : memref<16xf32>
    memref.store %1, %alloc_2[%c2] : memref<16xf32>
    memref.store %1, %alloc_2[%c3] : memref<16xf32>
    memref.store %1, %alloc_2[%c4] : memref<16xf32>
    memref.store %1, %alloc_2[%c5] : memref<16xf32>
    memref.store %1, %alloc_2[%c6] : memref<16xf32>
    memref.store %1, %alloc_2[%c7] : memref<16xf32>
    memref.store %1, %alloc_2[%c8] : memref<16xf32>
    memref.store %1, %alloc_2[%c9] : memref<16xf32>
    memref.store %1, %alloc_2[%c10] : memref<16xf32>
    memref.store %1, %alloc_2[%c11] : memref<16xf32>
    memref.store %1, %alloc_2[%c12] : memref<16xf32>
    memref.store %1, %alloc_2[%c13] : memref<16xf32>
    memref.store %1, %alloc_2[%c14] : memref<16xf32>
    memref.store %1, %alloc_2[%c15] : memref<16xf32>
    upmem.scatter %alloc_2[16384, 16, #map1] onto %2 : memref<16xf32> onto !upmem.hierarchy<4x64x16>
    %4 = memref.get_global @__constant_4096x256xf32 : memref<4096x256xf32>
    upmem.scatter %4[16448, 4096, #map6] onto %2 : memref<4096x256xf32> onto !upmem.hierarchy<4x64x16>
    upmem.launch_func  @dpu_kernels::@softmax_1048576 %2 : !upmem.hierarchy<4x64x16> 
    %alloc_3 = memref.alloc() : memref<4096x256xf32>
    upmem.gather %alloc_3[16448, 4096, #map6] from %2 : memref<4096x256xf32> from !upmem.hierarchy<4x64x16>
    %5 = memref.get_global @__constant_1xi64 : memref<1xi64>
    %reshape_4 = memref.reshape %alloc_3(%5) : (memref<4096x256xf32>, memref<1xi64>) -> memref<1048576xf32>
    upmem.free_dpus %2 : !upmem.hierarchy<4x64x16>
    %alloc_5 = memref.alloc() : memref<1048576xf32>
    scf.for %arg1 = %c0 to %c1048576 step %c1 {
      %9 = memref.load %reshape_4[%arg1] : memref<1048576xf32>
      %10 = llvm.intr.exp(%9)  : (f32) -> f32
      memref.store %10, %alloc_5[%arg1] : memref<1048576xf32>
    }
    %6 = memref.get_global @__constant_xf32_0 : memref<f32>
    %alloc_6 = memref.alloc() : memref<1024xf32>
    %alloc_7 = memref.alloc() : memref<f32>
    scf.for %arg1 = %c0 to %c1024 step %c1 {
      %9 = arith.muli %arg1, %c1024 : index
      %subview = memref.subview %alloc_5[%9] [1024] [1] : memref<1048576xf32> to memref<1024xf32, strided<[1], offset: ?>>
      memref.copy %6, %alloc_7 : memref<f32> to memref<f32>
      scf.for %arg2 = %c0 to %c1024 step %c1 {
        %11 = memref.load %subview[%arg2] : memref<1024xf32, strided<[1], offset: ?>>
        %12 = memref.load %alloc_7[] : memref<f32>
        %13 = arith.addf %11, %12 : f32
        memref.store %13, %alloc_7[] : memref<f32>
      }
      %10 = memref.load %alloc_7[] : memref<f32>
      memref.store %10, %alloc_6[%arg1] : memref<1024xf32>
    }
    %alloc_8 = memref.alloc() : memref<f32>
    memref.copy %6, %alloc_8 : memref<f32> to memref<f32>
    scf.for %arg1 = %c0 to %c1024 step %c1 {
      %9 = memref.load %alloc_6[%arg1] : memref<1024xf32>
      %10 = memref.load %alloc_8[] : memref<f32>
      %11 = arith.addf %9, %10 : f32
      memref.store %11, %alloc_8[] : memref<f32>
    }
    %7 = memref.load %alloc_8[] : memref<f32>
    %8 = upmem.alloc_dpus : !upmem.hierarchy<4x64x16>
    %reshape_9 = memref.reshape %alloc_5(%3) : (memref<1048576xf32>, memref<2xi64>) -> memref<4096x256xf32>
    upmem.scatter %reshape_9[0, 4096, #map6] onto %8 : memref<4096x256xf32> onto !upmem.hierarchy<4x64x16>
    %alloc_10 = memref.alloc() : memref<16xf32>
    memref.store %7, %alloc_10[%c0] : memref<16xf32>
    memref.store %7, %alloc_10[%c1] : memref<16xf32>
    memref.store %7, %alloc_10[%c2] : memref<16xf32>
    memref.store %7, %alloc_10[%c3] : memref<16xf32>
    memref.store %7, %alloc_10[%c4] : memref<16xf32>
    memref.store %7, %alloc_10[%c5] : memref<16xf32>
    memref.store %7, %alloc_10[%c6] : memref<16xf32>
    memref.store %7, %alloc_10[%c7] : memref<16xf32>
    memref.store %7, %alloc_10[%c8] : memref<16xf32>
    memref.store %7, %alloc_10[%c9] : memref<16xf32>
    memref.store %7, %alloc_10[%c10] : memref<16xf32>
    memref.store %7, %alloc_10[%c11] : memref<16xf32>
    memref.store %7, %alloc_10[%c12] : memref<16xf32>
    memref.store %7, %alloc_10[%c13] : memref<16xf32>
    memref.store %7, %alloc_10[%c14] : memref<16xf32>
    memref.store %7, %alloc_10[%c15] : memref<16xf32>
    upmem.scatter %alloc_10[16384, 16, #map1] onto %8 : memref<16xf32> onto !upmem.hierarchy<4x64x16>
    upmem.scatter %4[16448, 4096, #map6] onto %8 : memref<4096x256xf32> onto !upmem.hierarchy<4x64x16>
    upmem.launch_func  @dpu_kernels::@softmax_1048576_16 %8 : !upmem.hierarchy<4x64x16> 
    %alloc_11 = memref.alloc() : memref<4096x256xf32>
    upmem.gather %alloc_11[16448, 4096, #map6] from %8 : memref<4096x256xf32> from !upmem.hierarchy<4x64x16>
    %reshape_12 = memref.reshape %alloc_11(%5) : (memref<4096x256xf32>, memref<1xi64>) -> memref<1048576xf32>
    upmem.free_dpus %8 : !upmem.hierarchy<4x64x16>
    return %reshape_12 : memref<1048576xf32>
  }
  func.func @va_1048576(%arg0: memref<1048576xf32>, %arg1: memref<1048576xf32>) -> memref<1048576xf32> {
    %0 = upmem.alloc_dpus : !upmem.hierarchy<4x64x16>
    %1 = memref.get_global @__constant_2xi64 : memref<2xi64>
    %reshape = memref.reshape %arg0(%1) : (memref<1048576xf32>, memref<2xi64>) -> memref<4096x256xf32>
    upmem.scatter %reshape[0, 4096, #map6] onto %0 : memref<4096x256xf32> onto !upmem.hierarchy<4x64x16>
    %reshape_0 = memref.reshape %arg1(%1) : (memref<1048576xf32>, memref<2xi64>) -> memref<4096x256xf32>
    upmem.scatter %reshape_0[16384, 4096, #map6] onto %0 : memref<4096x256xf32> onto !upmem.hierarchy<4x64x16>
    %2 = memref.get_global @__constant_4096x256xf32 : memref<4096x256xf32>
    upmem.scatter %2[32768, 4096, #map6] onto %0 : memref<4096x256xf32> onto !upmem.hierarchy<4x64x16>
    upmem.launch_func  @dpu_kernels::@va_1048576 %0 : !upmem.hierarchy<4x64x16> 
    %alloc = memref.alloc() : memref<4096x256xf32>
    upmem.gather %alloc[32768, 4096, #map6] from %0 : memref<4096x256xf32> from !upmem.hierarchy<4x64x16>
    %3 = memref.get_global @__constant_1xi64 : memref<1xi64>
    %reshape_1 = memref.reshape %alloc(%3) : (memref<4096x256xf32>, memref<1xi64>) -> memref<1048576xf32>
    upmem.free_dpus %0 : !upmem.hierarchy<4x64x16>
    return %reshape_1 : memref<1048576xf32>
  }
  func.func @rmsnorm_262144(%arg0: memref<262144xf32>, %arg1: memref<262144xf32>) -> memref<262144xf32> {
    %c256 = arith.constant 256 : index
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
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 9.99999974E-6 : f32
    %cst_0 = arith.constant 2.621440e+05 : f32
    %0 = upmem.alloc_dpus : !upmem.hierarchy<4x64x16>
    %1 = memref.get_global @__constant_2xi64_0 : memref<2xi64>
    %reshape = memref.reshape %arg0(%1) : (memref<262144xf32>, memref<2xi64>) -> memref<4096x64xf32>
    upmem.scatter %reshape[0, 1024, #map6] onto %0 : memref<4096x64xf32> onto !upmem.hierarchy<4x64x16>
    upmem.scatter %reshape[4096, 1024, #map6] onto %0 : memref<4096x64xf32> onto !upmem.hierarchy<4x64x16>
    %2 = memref.get_global @__constant_4096x64xf32 : memref<4096x64xf32>
    upmem.scatter %2[8192, 1024, #map6] onto %0 : memref<4096x64xf32> onto !upmem.hierarchy<4x64x16>
    upmem.launch_func  @dpu_kernels::@rmsnorm_262144 %0 : !upmem.hierarchy<4x64x16> 
    %alloc = memref.alloc() : memref<4096x64xf32>
    upmem.gather %alloc[8192, 1024, #map6] from %0 : memref<4096x64xf32> from !upmem.hierarchy<4x64x16>
    %3 = memref.get_global @__constant_1xi64_0 : memref<1xi64>
    %reshape_1 = memref.reshape %alloc(%3) : (memref<4096x64xf32>, memref<1xi64>) -> memref<262144xf32>
    upmem.free_dpus %0 : !upmem.hierarchy<4x64x16>
    %4 = memref.get_global @__constant_xf32_0 : memref<f32>
    %alloc_2 = memref.alloc() : memref<256xf32>
    %alloc_3 = memref.alloc() : memref<f32>
    scf.for %arg2 = %c0 to %c256 step %c1 {
      %13 = arith.muli %arg2, %c1024 : index
      %subview = memref.subview %reshape_1[%13] [1024] [1] : memref<262144xf32> to memref<1024xf32, strided<[1], offset: ?>>
      memref.copy %4, %alloc_3 : memref<f32> to memref<f32>
      scf.for %arg3 = %c0 to %c1024 step %c1 {
        %15 = memref.load %subview[%arg3] : memref<1024xf32, strided<[1], offset: ?>>
        %16 = memref.load %alloc_3[] : memref<f32>
        %17 = arith.addf %15, %16 : f32
        memref.store %17, %alloc_3[] : memref<f32>
      }
      %14 = memref.load %alloc_3[] : memref<f32>
      memref.store %14, %alloc_2[%arg2] : memref<256xf32>
    }
    %alloc_4 = memref.alloc() : memref<f32>
    memref.copy %4, %alloc_4 : memref<f32> to memref<f32>
    scf.for %arg2 = %c0 to %c256 step %c1 {
      %13 = memref.load %alloc_2[%arg2] : memref<256xf32>
      %14 = memref.load %alloc_4[] : memref<f32>
      %15 = arith.addf %13, %14 : f32
      memref.store %15, %alloc_4[] : memref<f32>
    }
    %5 = memref.load %alloc_4[] : memref<f32>
    %6 = arith.divf %5, %cst_0 : f32
    %7 = arith.addf %6, %cst : f32
    %8 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %9 = llvm.intr.sqrt(%7)  : (f32) -> f32
    %10 = llvm.fdiv %8, %9  : f32
    %11 = upmem.alloc_dpus : !upmem.hierarchy<4x64x16>
    upmem.scatter %reshape[0, 1024, #map6] onto %11 : memref<4096x64xf32> onto !upmem.hierarchy<4x64x16>
    %alloc_5 = memref.alloc() : memref<16xf32>
    memref.store %10, %alloc_5[%c0] : memref<16xf32>
    memref.store %10, %alloc_5[%c1] : memref<16xf32>
    memref.store %10, %alloc_5[%c2] : memref<16xf32>
    memref.store %10, %alloc_5[%c3] : memref<16xf32>
    memref.store %10, %alloc_5[%c4] : memref<16xf32>
    memref.store %10, %alloc_5[%c5] : memref<16xf32>
    memref.store %10, %alloc_5[%c6] : memref<16xf32>
    memref.store %10, %alloc_5[%c7] : memref<16xf32>
    memref.store %10, %alloc_5[%c8] : memref<16xf32>
    memref.store %10, %alloc_5[%c9] : memref<16xf32>
    memref.store %10, %alloc_5[%c10] : memref<16xf32>
    memref.store %10, %alloc_5[%c11] : memref<16xf32>
    memref.store %10, %alloc_5[%c12] : memref<16xf32>
    memref.store %10, %alloc_5[%c13] : memref<16xf32>
    memref.store %10, %alloc_5[%c14] : memref<16xf32>
    memref.store %10, %alloc_5[%c15] : memref<16xf32>
    upmem.scatter %alloc_5[4096, 16, #map1] onto %11 : memref<16xf32> onto !upmem.hierarchy<4x64x16>
    upmem.scatter %2[4160, 1024, #map6] onto %11 : memref<4096x64xf32> onto !upmem.hierarchy<4x64x16>
    upmem.launch_func  @dpu_kernels::@rmsnorm_262144_17 %11 : !upmem.hierarchy<4x64x16> 
    %alloc_6 = memref.alloc() : memref<4096x64xf32>
    upmem.gather %alloc_6[4160, 1024, #map6] from %11 : memref<4096x64xf32> from !upmem.hierarchy<4x64x16>
    %reshape_7 = memref.reshape %alloc_6(%3) : (memref<4096x64xf32>, memref<1xi64>) -> memref<262144xf32>
    upmem.free_dpus %11 : !upmem.hierarchy<4x64x16>
    %12 = upmem.alloc_dpus : !upmem.hierarchy<4x64x16>
    %reshape_8 = memref.reshape %reshape_7(%1) : (memref<262144xf32>, memref<2xi64>) -> memref<4096x64xf32>
    upmem.scatter %reshape_8[0, 1024, #map6] onto %12 : memref<4096x64xf32> onto !upmem.hierarchy<4x64x16>
    %reshape_9 = memref.reshape %arg1(%1) : (memref<262144xf32>, memref<2xi64>) -> memref<4096x64xf32>
    upmem.scatter %reshape_9[4096, 1024, #map6] onto %12 : memref<4096x64xf32> onto !upmem.hierarchy<4x64x16>
    upmem.scatter %2[8192, 1024, #map6] onto %12 : memref<4096x64xf32> onto !upmem.hierarchy<4x64x16>
    upmem.launch_func  @dpu_kernels::@rmsnorm_262144 %12 : !upmem.hierarchy<4x64x16> 
    %alloc_10 = memref.alloc() : memref<4096x64xf32>
    upmem.gather %alloc_10[8192, 1024, #map6] from %12 : memref<4096x64xf32> from !upmem.hierarchy<4x64x16>
    %reshape_11 = memref.reshape %alloc_10(%3) : (memref<4096x64xf32>, memref<1xi64>) -> memref<262144xf32>
    upmem.free_dpus %12 : !upmem.hierarchy<4x64x16>
    return %reshape_11 : memref<262144xf32>
  }
  func.func @softmax_262144(%arg0: memref<262144xf32>) -> memref<262144xf32> {
    %c262144 = arith.constant 262144 : index
    %c256 = arith.constant 256 : index
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
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_xf32 : memref<f32>
    %alloc = memref.alloc() : memref<256xf32>
    %alloc_0 = memref.alloc() : memref<f32>
    scf.for %arg1 = %c0 to %c256 step %c1 {
      %9 = arith.muli %arg1, %c1024 : index
      %subview = memref.subview %arg0[%9] [1024] [1] : memref<262144xf32> to memref<1024xf32, strided<[1], offset: ?>>
      memref.copy %0, %alloc_0 : memref<f32> to memref<f32>
      scf.for %arg2 = %c0 to %c1024 step %c1 {
        %11 = memref.load %subview[%arg2] : memref<1024xf32, strided<[1], offset: ?>>
        %12 = memref.load %alloc_0[] : memref<f32>
        %13 = arith.maximumf %11, %12 : f32
        memref.store %13, %alloc_0[] : memref<f32>
      }
      %10 = memref.load %alloc_0[] : memref<f32>
      memref.store %10, %alloc[%arg1] : memref<256xf32>
    }
    %alloc_1 = memref.alloc() : memref<f32>
    memref.copy %0, %alloc_1 : memref<f32> to memref<f32>
    scf.for %arg1 = %c0 to %c256 step %c1 {
      %9 = memref.load %alloc[%arg1] : memref<256xf32>
      %10 = memref.load %alloc_1[] : memref<f32>
      %11 = arith.maximumf %9, %10 : f32
      memref.store %11, %alloc_1[] : memref<f32>
    }
    %1 = memref.load %alloc_1[] : memref<f32>
    %2 = upmem.alloc_dpus : !upmem.hierarchy<4x64x16>
    %3 = memref.get_global @__constant_2xi64_0 : memref<2xi64>
    %reshape = memref.reshape %arg0(%3) : (memref<262144xf32>, memref<2xi64>) -> memref<4096x64xf32>
    upmem.scatter %reshape[0, 1024, #map6] onto %2 : memref<4096x64xf32> onto !upmem.hierarchy<4x64x16>
    %alloc_2 = memref.alloc() : memref<16xf32>
    memref.store %1, %alloc_2[%c0] : memref<16xf32>
    memref.store %1, %alloc_2[%c1] : memref<16xf32>
    memref.store %1, %alloc_2[%c2] : memref<16xf32>
    memref.store %1, %alloc_2[%c3] : memref<16xf32>
    memref.store %1, %alloc_2[%c4] : memref<16xf32>
    memref.store %1, %alloc_2[%c5] : memref<16xf32>
    memref.store %1, %alloc_2[%c6] : memref<16xf32>
    memref.store %1, %alloc_2[%c7] : memref<16xf32>
    memref.store %1, %alloc_2[%c8] : memref<16xf32>
    memref.store %1, %alloc_2[%c9] : memref<16xf32>
    memref.store %1, %alloc_2[%c10] : memref<16xf32>
    memref.store %1, %alloc_2[%c11] : memref<16xf32>
    memref.store %1, %alloc_2[%c12] : memref<16xf32>
    memref.store %1, %alloc_2[%c13] : memref<16xf32>
    memref.store %1, %alloc_2[%c14] : memref<16xf32>
    memref.store %1, %alloc_2[%c15] : memref<16xf32>
    upmem.scatter %alloc_2[4096, 16, #map1] onto %2 : memref<16xf32> onto !upmem.hierarchy<4x64x16>
    %4 = memref.get_global @__constant_4096x64xf32 : memref<4096x64xf32>
    upmem.scatter %4[4160, 1024, #map6] onto %2 : memref<4096x64xf32> onto !upmem.hierarchy<4x64x16>
    upmem.launch_func  @dpu_kernels::@softmax_262144 %2 : !upmem.hierarchy<4x64x16> 
    %alloc_3 = memref.alloc() : memref<4096x64xf32>
    upmem.gather %alloc_3[4160, 1024, #map6] from %2 : memref<4096x64xf32> from !upmem.hierarchy<4x64x16>
    %5 = memref.get_global @__constant_1xi64_0 : memref<1xi64>
    %reshape_4 = memref.reshape %alloc_3(%5) : (memref<4096x64xf32>, memref<1xi64>) -> memref<262144xf32>
    upmem.free_dpus %2 : !upmem.hierarchy<4x64x16>
    %alloc_5 = memref.alloc() : memref<262144xf32>
    scf.for %arg1 = %c0 to %c262144 step %c1 {
      %9 = memref.load %reshape_4[%arg1] : memref<262144xf32>
      %10 = llvm.intr.exp(%9)  : (f32) -> f32
      memref.store %10, %alloc_5[%arg1] : memref<262144xf32>
    }
    %6 = memref.get_global @__constant_xf32_0 : memref<f32>
    %alloc_6 = memref.alloc() : memref<256xf32>
    %alloc_7 = memref.alloc() : memref<f32>
    scf.for %arg1 = %c0 to %c256 step %c1 {
      %9 = arith.muli %arg1, %c1024 : index
      %subview = memref.subview %alloc_5[%9] [1024] [1] : memref<262144xf32> to memref<1024xf32, strided<[1], offset: ?>>
      memref.copy %6, %alloc_7 : memref<f32> to memref<f32>
      scf.for %arg2 = %c0 to %c1024 step %c1 {
        %11 = memref.load %subview[%arg2] : memref<1024xf32, strided<[1], offset: ?>>
        %12 = memref.load %alloc_7[] : memref<f32>
        %13 = arith.addf %11, %12 : f32
        memref.store %13, %alloc_7[] : memref<f32>
      }
      %10 = memref.load %alloc_7[] : memref<f32>
      memref.store %10, %alloc_6[%arg1] : memref<256xf32>
    }
    %alloc_8 = memref.alloc() : memref<f32>
    memref.copy %6, %alloc_8 : memref<f32> to memref<f32>
    scf.for %arg1 = %c0 to %c256 step %c1 {
      %9 = memref.load %alloc_6[%arg1] : memref<256xf32>
      %10 = memref.load %alloc_8[] : memref<f32>
      %11 = arith.addf %9, %10 : f32
      memref.store %11, %alloc_8[] : memref<f32>
    }
    %7 = memref.load %alloc_8[] : memref<f32>
    %8 = upmem.alloc_dpus : !upmem.hierarchy<4x64x16>
    %reshape_9 = memref.reshape %alloc_5(%3) : (memref<262144xf32>, memref<2xi64>) -> memref<4096x64xf32>
    upmem.scatter %reshape_9[0, 1024, #map6] onto %8 : memref<4096x64xf32> onto !upmem.hierarchy<4x64x16>
    %alloc_10 = memref.alloc() : memref<16xf32>
    memref.store %7, %alloc_10[%c0] : memref<16xf32>
    memref.store %7, %alloc_10[%c1] : memref<16xf32>
    memref.store %7, %alloc_10[%c2] : memref<16xf32>
    memref.store %7, %alloc_10[%c3] : memref<16xf32>
    memref.store %7, %alloc_10[%c4] : memref<16xf32>
    memref.store %7, %alloc_10[%c5] : memref<16xf32>
    memref.store %7, %alloc_10[%c6] : memref<16xf32>
    memref.store %7, %alloc_10[%c7] : memref<16xf32>
    memref.store %7, %alloc_10[%c8] : memref<16xf32>
    memref.store %7, %alloc_10[%c9] : memref<16xf32>
    memref.store %7, %alloc_10[%c10] : memref<16xf32>
    memref.store %7, %alloc_10[%c11] : memref<16xf32>
    memref.store %7, %alloc_10[%c12] : memref<16xf32>
    memref.store %7, %alloc_10[%c13] : memref<16xf32>
    memref.store %7, %alloc_10[%c14] : memref<16xf32>
    memref.store %7, %alloc_10[%c15] : memref<16xf32>
    upmem.scatter %alloc_10[4096, 16, #map1] onto %8 : memref<16xf32> onto !upmem.hierarchy<4x64x16>
    upmem.scatter %4[4160, 1024, #map6] onto %8 : memref<4096x64xf32> onto !upmem.hierarchy<4x64x16>
    upmem.launch_func  @dpu_kernels::@softmax_262144_19 %8 : !upmem.hierarchy<4x64x16> 
    %alloc_11 = memref.alloc() : memref<4096x64xf32>
    upmem.gather %alloc_11[4160, 1024, #map6] from %8 : memref<4096x64xf32> from !upmem.hierarchy<4x64x16>
    %reshape_12 = memref.reshape %alloc_11(%5) : (memref<4096x64xf32>, memref<1xi64>) -> memref<262144xf32>
    upmem.free_dpus %8 : !upmem.hierarchy<4x64x16>
    return %reshape_12 : memref<262144xf32>
  }
  func.func @rmsnorm_262144_opt(%arg0: memref<262144xf32>, %arg1: memref<262144xf32>) -> memref<262144xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index
    %c8 = arith.constant 8 : index
    %c9 = arith.constant 9 : index
    %c10 = arith.constant 10 : index
    %c11 = arith.constant 11 : index
    %c12 = arith.constant 12 : index
    %c13 = arith.constant 13 : index
    %c14 = arith.constant 14 : index
    %c15 = arith.constant 15 : index
    %c4096 = arith.constant 4096 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 9.99999974E-6 : f32
    %cst_1 = arith.constant 2.621440e+05 : f32
    %0 = memref.get_global @__constant_2xi64_0 : memref<2xi64>
    %reshape = memref.reshape %arg0(%0) : (memref<262144xf32>, memref<2xi64>) -> memref<4096x64xf32>
    %1 = upmem.alloc_dpus : !upmem.hierarchy<4x64x16>
    upmem.scatter %reshape[0, 1024, #map6] onto %1 : memref<4096x64xf32> onto !upmem.hierarchy<4x64x16>
    upmem.scatter %reshape[4096, 1024, #map6] onto %1 : memref<4096x64xf32> onto !upmem.hierarchy<4x64x16>
    upmem.launch_func  @dpu_kernels::@rmsnorm_262144_opt %1 : !upmem.hierarchy<4x64x16> 
    %alloc = memref.alloc() : memref<4096x2xf32>
    upmem.gather %alloc[8192, 32, #map6] from %1 : memref<4096x2xf32> from !upmem.hierarchy<4x64x16>
    upmem.free_dpus %1 : !upmem.hierarchy<4x64x16>
    %2 = scf.for %arg2 = %c0 to %c4096 step %c1 iter_args(%arg3 = %cst) -> (f32) {
      %11 = memref.load %alloc[%arg2, %c0] : memref<4096x2xf32>
      %12 = memref.load %alloc[%arg2, %c1] : memref<4096x2xf32>
      %13 = arith.addf %11, %12 : f32
      %14 = arith.addf %13, %arg3 : f32
      scf.yield %14 : f32
    }
    %3 = arith.divf %2, %cst_1 : f32
    %4 = arith.addf %3, %cst_0 : f32
    %5 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %6 = llvm.intr.sqrt(%4)  : (f32) -> f32
    %7 = llvm.fdiv %5, %6  : f32
    %alloc_2 = memref.alloc() : memref<16xf32>
    memref.store %7, %alloc_2[%c0] : memref<16xf32>
    memref.store %7, %alloc_2[%c1] : memref<16xf32>
    memref.store %7, %alloc_2[%c2] : memref<16xf32>
    memref.store %7, %alloc_2[%c3] : memref<16xf32>
    memref.store %7, %alloc_2[%c4] : memref<16xf32>
    memref.store %7, %alloc_2[%c5] : memref<16xf32>
    memref.store %7, %alloc_2[%c6] : memref<16xf32>
    memref.store %7, %alloc_2[%c7] : memref<16xf32>
    memref.store %7, %alloc_2[%c8] : memref<16xf32>
    memref.store %7, %alloc_2[%c9] : memref<16xf32>
    memref.store %7, %alloc_2[%c10] : memref<16xf32>
    memref.store %7, %alloc_2[%c11] : memref<16xf32>
    memref.store %7, %alloc_2[%c12] : memref<16xf32>
    memref.store %7, %alloc_2[%c13] : memref<16xf32>
    memref.store %7, %alloc_2[%c14] : memref<16xf32>
    memref.store %7, %alloc_2[%c15] : memref<16xf32>
    %8 = upmem.alloc_dpus : !upmem.hierarchy<4x64x16>
    upmem.scatter %reshape[0, 1024, #map6] onto %8 : memref<4096x64xf32> onto !upmem.hierarchy<4x64x16>
    %reshape_3 = memref.reshape %arg1(%0) : (memref<262144xf32>, memref<2xi64>) -> memref<4096x64xf32>
    upmem.scatter %reshape_3[4096, 1024, #map6] onto %8 : memref<4096x64xf32> onto !upmem.hierarchy<4x64x16>
    upmem.scatter %alloc_2[8192, 16, #map1] onto %8 : memref<16xf32> onto !upmem.hierarchy<4x64x16>
    %9 = memref.get_global @__constant_4096x64xf32 : memref<4096x64xf32>
    upmem.scatter %9[8256, 1024, #map6] onto %8 : memref<4096x64xf32> onto !upmem.hierarchy<4x64x16>
    upmem.launch_func  @dpu_kernels::@rmsnorm_262144_opt_20 %8 : !upmem.hierarchy<4x64x16> 
    %alloc_4 = memref.alloc() : memref<4096x64xf32>
    upmem.gather %alloc_4[8256, 1024, #map6] from %8 : memref<4096x64xf32> from !upmem.hierarchy<4x64x16>
    %10 = memref.get_global @__constant_1xi64_0 : memref<1xi64>
    %reshape_5 = memref.reshape %alloc_4(%10) : (memref<4096x64xf32>, memref<1xi64>) -> memref<262144xf32>
    upmem.free_dpus %8 : !upmem.hierarchy<4x64x16>
    return %reshape_5 : memref<262144xf32>
  }
  func.func @softmax_262144_opt(%arg0: memref<262144xf32>) -> memref<262144xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index
    %c8 = arith.constant 8 : index
    %c9 = arith.constant 9 : index
    %c10 = arith.constant 10 : index
    %c11 = arith.constant 11 : index
    %c12 = arith.constant 12 : index
    %c13 = arith.constant 13 : index
    %c14 = arith.constant 14 : index
    %c15 = arith.constant 15 : index
    %c4096 = arith.constant 4096 : index
    %c262144 = arith.constant 262144 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %0 = scf.for %arg1 = %c0 to %c262144 step %c1 iter_args(%arg2 = %cst_0) -> (f32) {
      %8 = memref.load %arg0[%arg1] : memref<262144xf32>
      %9 = arith.maximumf %8, %arg2 : f32
      scf.yield %9 : f32
    }
    %alloc = memref.alloc() : memref<16xf32>
    memref.store %0, %alloc[%c0] : memref<16xf32>
    memref.store %0, %alloc[%c1] : memref<16xf32>
    memref.store %0, %alloc[%c2] : memref<16xf32>
    memref.store %0, %alloc[%c3] : memref<16xf32>
    memref.store %0, %alloc[%c4] : memref<16xf32>
    memref.store %0, %alloc[%c5] : memref<16xf32>
    memref.store %0, %alloc[%c6] : memref<16xf32>
    memref.store %0, %alloc[%c7] : memref<16xf32>
    memref.store %0, %alloc[%c8] : memref<16xf32>
    memref.store %0, %alloc[%c9] : memref<16xf32>
    memref.store %0, %alloc[%c10] : memref<16xf32>
    memref.store %0, %alloc[%c11] : memref<16xf32>
    memref.store %0, %alloc[%c12] : memref<16xf32>
    memref.store %0, %alloc[%c13] : memref<16xf32>
    memref.store %0, %alloc[%c14] : memref<16xf32>
    memref.store %0, %alloc[%c15] : memref<16xf32>
    %1 = memref.get_global @__constant_2xi64_0 : memref<2xi64>
    %reshape = memref.reshape %arg0(%1) : (memref<262144xf32>, memref<2xi64>) -> memref<4096x64xf32>
    %2 = upmem.alloc_dpus : !upmem.hierarchy<4x64x16>
    upmem.scatter %reshape[0, 1024, #map6] onto %2 : memref<4096x64xf32> onto !upmem.hierarchy<4x64x16>
    upmem.scatter %alloc[4096, 16, #map1] onto %2 : memref<16xf32> onto !upmem.hierarchy<4x64x16>
    %3 = memref.get_global @__constant_4096x64xf32 : memref<4096x64xf32>
    upmem.scatter %3[4160, 1024, #map6] onto %2 : memref<4096x64xf32> onto !upmem.hierarchy<4x64x16>
    %4 = memref.get_global @__constant_4096x2xf32 : memref<4096x2xf32>
    upmem.scatter %4[8256, 32, #map6] onto %2 : memref<4096x2xf32> onto !upmem.hierarchy<4x64x16>
    upmem.launch_func  @dpu_kernels::@softmax_262144_opt %2 : !upmem.hierarchy<4x64x16> 
    %alloc_1 = memref.alloc() : memref<4096x2xf32>
    upmem.gather %alloc_1[8256, 32, #map6] from %2 : memref<4096x2xf32> from !upmem.hierarchy<4x64x16>
    %5 = scf.for %arg1 = %c0 to %c4096 step %c1 iter_args(%arg2 = %cst) -> (f32) {
      %8 = memref.load %alloc_1[%arg1, %c0] : memref<4096x2xf32>
      %9 = memref.load %alloc_1[%arg1, %c1] : memref<4096x2xf32>
      %10 = arith.addf %8, %9 : f32
      %11 = arith.addf %10, %arg2 : f32
      scf.yield %11 : f32
    }
    %alloc_2 = memref.alloc() : memref<16xf32>
    memref.store %5, %alloc_2[%c0] : memref<16xf32>
    memref.store %5, %alloc_2[%c1] : memref<16xf32>
    memref.store %5, %alloc_2[%c2] : memref<16xf32>
    memref.store %5, %alloc_2[%c3] : memref<16xf32>
    memref.store %5, %alloc_2[%c4] : memref<16xf32>
    memref.store %5, %alloc_2[%c5] : memref<16xf32>
    memref.store %5, %alloc_2[%c6] : memref<16xf32>
    memref.store %5, %alloc_2[%c7] : memref<16xf32>
    memref.store %5, %alloc_2[%c8] : memref<16xf32>
    memref.store %5, %alloc_2[%c9] : memref<16xf32>
    memref.store %5, %alloc_2[%c10] : memref<16xf32>
    memref.store %5, %alloc_2[%c11] : memref<16xf32>
    memref.store %5, %alloc_2[%c12] : memref<16xf32>
    memref.store %5, %alloc_2[%c13] : memref<16xf32>
    memref.store %5, %alloc_2[%c14] : memref<16xf32>
    memref.store %5, %alloc_2[%c15] : memref<16xf32>
    %alloc_3 = memref.alloc() : memref<4096x64xf32>
    upmem.gather %alloc_3[4160, 1024, #map6] from %2 : memref<4096x64xf32> from !upmem.hierarchy<4x64x16>
    upmem.free_dpus %2 : !upmem.hierarchy<4x64x16>
    %6 = upmem.alloc_dpus : !upmem.hierarchy<4x64x16>
    upmem.scatter %alloc_3[0, 1024, #map6] onto %6 : memref<4096x64xf32> onto !upmem.hierarchy<4x64x16>
    upmem.scatter %alloc_2[4096, 16, #map1] onto %6 : memref<16xf32> onto !upmem.hierarchy<4x64x16>
    upmem.scatter %3[4160, 1024, #map6] onto %6 : memref<4096x64xf32> onto !upmem.hierarchy<4x64x16>
    upmem.launch_func  @dpu_kernels::@softmax_262144_opt_21 %6 : !upmem.hierarchy<4x64x16> 
    %alloc_4 = memref.alloc() : memref<4096x64xf32>
    upmem.gather %alloc_4[4160, 1024, #map6] from %6 : memref<4096x64xf32> from !upmem.hierarchy<4x64x16>
    %7 = memref.get_global @__constant_1xi64_0 : memref<1xi64>
    %reshape_5 = memref.reshape %alloc_4(%7) : (memref<4096x64xf32>, memref<1xi64>) -> memref<262144xf32>
    upmem.free_dpus %6 : !upmem.hierarchy<4x64x16>
    return %reshape_5 : memref<262144xf32>
  }
  func.func @va_262144(%arg0: memref<262144xf32>, %arg1: memref<262144xf32>) -> memref<262144xf32> {
    %0 = upmem.alloc_dpus : !upmem.hierarchy<4x64x16>
    %1 = memref.get_global @__constant_2xi64_0 : memref<2xi64>
    %reshape = memref.reshape %arg0(%1) : (memref<262144xf32>, memref<2xi64>) -> memref<4096x64xf32>
    upmem.scatter %reshape[0, 1024, #map6] onto %0 : memref<4096x64xf32> onto !upmem.hierarchy<4x64x16>
    %reshape_0 = memref.reshape %arg1(%1) : (memref<262144xf32>, memref<2xi64>) -> memref<4096x64xf32>
    upmem.scatter %reshape_0[4096, 1024, #map6] onto %0 : memref<4096x64xf32> onto !upmem.hierarchy<4x64x16>
    %2 = memref.get_global @__constant_4096x64xf32 : memref<4096x64xf32>
    upmem.scatter %2[8192, 1024, #map6] onto %0 : memref<4096x64xf32> onto !upmem.hierarchy<4x64x16>
    upmem.launch_func  @dpu_kernels::@va_262144 %0 : !upmem.hierarchy<4x64x16> 
    %alloc = memref.alloc() : memref<4096x64xf32>
    upmem.gather %alloc[8192, 1024, #map6] from %0 : memref<4096x64xf32> from !upmem.hierarchy<4x64x16>
    %3 = memref.get_global @__constant_1xi64_0 : memref<1xi64>
    %reshape_1 = memref.reshape %alloc(%3) : (memref<4096x64xf32>, memref<1xi64>) -> memref<262144xf32>
    upmem.free_dpus %0 : !upmem.hierarchy<4x64x16>
    return %reshape_1 : memref<262144xf32>
  }
  func.func @mha_big(%arg0: memref<32768xf32>, %arg1: memref<1024x32768xf32>, %arg2: memref<1024x32768xf32>, %arg3: index) -> memref<32768xf32> {
    %c15 = arith.constant 15 : index
    %c14 = arith.constant 14 : index
    %c13 = arith.constant 13 : index
    %c12 = arith.constant 12 : index
    %c11 = arith.constant 11 : index
    %c10 = arith.constant 10 : index
    %c9 = arith.constant 9 : index
    %c7 = arith.constant 7 : index
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c4096 = arith.constant 4096 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 6.92820311 : f32
    %cst_1 = arith.constant 0xFF800000 : f32
    %0 = arith.addi %arg3, %c1 : index
    %alloc = memref.alloc() : memref<1024xf32>
    scf.for %arg4 = %c0 to %c1024 step %c1 {
      memref.store %cst_1, %alloc[%arg4] : memref<1024xf32>
    }
    %alloc_2 = memref.alloc() : memref<32768xf32>
    %alloc_3 = memref.alloc() : memref<32768xf32>
    memref.copy %alloc_2, %alloc_3 : memref<32768xf32> to memref<32768xf32>
    %alloc_4 = memref.alloc() : memref<1024xf32>
    %alloc_5 = memref.alloc() : memref<4096xf32>
    %alloc_6 = memref.alloc() : memref<4096xf32>
    %alloc_7 = memref.alloc() : memref<256x16xf32>
    %alloc_8 = memref.alloc() : memref<4xf32>
    %alloc_9 = memref.alloc() : memref<f32>
    %alloc_10 = memref.alloc() : memref<f32>
    %alloc_11 = memref.alloc() : memref<1xf32>
    %alloc_12 = memref.alloc() : memref<f32>
    %alloc_13 = memref.alloc() : memref<f32>
    %alloc_14 = memref.alloc() : memref<16xf32>
    %alloc_15 = memref.alloc() : memref<256x4xf32>
    %alloc_16 = memref.alloc() : memref<1024xf32>
    %alloc_17 = memref.alloc() : memref<1xf32>
    %alloc_18 = memref.alloc() : memref<f32>
    %alloc_19 = memref.alloc() : memref<f32>
    %alloc_20 = memref.alloc() : memref<16xf32>
    %alloc_21 = memref.alloc() : memref<256x4xf32>
    %alloc_22 = memref.alloc() : memref<4096xf32>
    %alloc_23 = memref.alloc() : memref<4096xf32>
    %alloc_24 = memref.alloc() : memref<4096xf32>
    %alloc_25 = memref.alloc() : memref<16xf32>
    %alloc_26 = memref.alloc() : memref<256x16xf32>
    %1 = scf.for %arg4 = %c0 to %c8 step %c1 iter_args(%arg5 = %alloc_3) -> (memref<32768xf32>) {
      %2 = arith.muli %arg4, %c4096 : index
      memref.copy %alloc, %alloc_4 : memref<1024xf32> to memref<1024xf32>
      %3 = scf.for %arg6 = %c0 to %0 step %c1 iter_args(%arg7 = %alloc_4) -> (memref<1024xf32>) {
        %subview_31 = memref.subview %arg0[%2] [4096] [1] : memref<32768xf32> to memref<4096xf32, strided<[1], offset: ?>>
        %subview_32 = memref.subview %arg1[%arg6, %2] [1, 4096] [1, 1] : memref<1024x32768xf32> to memref<4096xf32, strided<[1], offset: ?>>
        %22 = upmem.alloc_dpus : !upmem.hierarchy<1x16x16>
        %23 = memref.get_global @__constant_2xi64_4 : memref<2xi64>
        memref.copy %subview_31, %alloc_5 : memref<4096xf32, strided<[1], offset: ?>> to memref<4096xf32>
        %reshape_33 = memref.reshape %alloc_5(%23) : (memref<4096xf32>, memref<2xi64>) -> memref<256x16xf32>
        upmem.scatter %reshape_33[0, 256, #map3] onto %22 : memref<256x16xf32> onto !upmem.hierarchy<1x16x16>
        memref.copy %subview_32, %alloc_6 : memref<4096xf32, strided<[1], offset: ?>> to memref<4096xf32>
        %reshape_34 = memref.reshape %alloc_6(%23) : (memref<4096xf32>, memref<2xi64>) -> memref<256x16xf32>
        upmem.scatter %reshape_34[1024, 256, #map3] onto %22 : memref<256x16xf32> onto !upmem.hierarchy<1x16x16>
        %24 = memref.get_global @__constant_256x16xf32 : memref<256x16xf32>
        upmem.scatter %24[2048, 256, #map3] onto %22 : memref<256x16xf32> onto !upmem.hierarchy<1x16x16>
        upmem.launch_func  @dpu_kernels::@mha_big %22 : !upmem.hierarchy<1x16x16> 
        upmem.gather %alloc_7[2048, 256, #map3] from %22 : memref<256x16xf32> from !upmem.hierarchy<1x16x16>
        %25 = memref.get_global @__constant_1xi64_4 : memref<1xi64>
        %reshape_35 = memref.reshape %alloc_7(%25) : (memref<256x16xf32>, memref<1xi64>) -> memref<4096xf32>
        upmem.free_dpus %22 : !upmem.hierarchy<1x16x16>
        %26 = memref.get_global @__constant_xf32_0 : memref<f32>
        scf.for %arg8 = %c0 to %c4 step %c1 {
          %29 = arith.muli %arg8, %c1024 : index
          %subview_37 = memref.subview %reshape_35[%29] [1024] [1] : memref<4096xf32> to memref<1024xf32, strided<[1], offset: ?>>
          memref.copy %26, %alloc_9 : memref<f32> to memref<f32>
          scf.for %arg9 = %c0 to %c1024 step %c1 {
            %31 = memref.load %subview_37[%arg9] : memref<1024xf32, strided<[1], offset: ?>>
            %32 = memref.load %alloc_9[] : memref<f32>
            %33 = arith.addf %31, %32 : f32
            memref.store %33, %alloc_9[] : memref<f32>
          }
          %30 = memref.load %alloc_9[] : memref<f32>
          memref.store %30, %alloc_8[%arg8] : memref<4xf32>
        }
        memref.copy %26, %alloc_10 : memref<f32> to memref<f32>
        scf.for %arg8 = %c0 to %c4 step %c1 {
          %29 = memref.load %alloc_8[%arg8] : memref<4xf32>
          %30 = memref.load %alloc_10[] : memref<f32>
          %31 = arith.addf %29, %30 : f32
          memref.store %31, %alloc_10[] : memref<f32>
        }
        %27 = memref.load %alloc_10[] : memref<f32>
        %28 = arith.divf %27, %cst_0 : f32
        %alloc_36 = memref.alloc() : memref<1024xf32>
        memref.copy %arg7, %alloc_36 : memref<1024xf32> to memref<1024xf32>
        memref.store %28, %alloc_36[%arg6] : memref<1024xf32>
        scf.yield %alloc_36 : memref<1024xf32>
      }
      %4 = memref.get_global @__constant_xf32 : memref<f32>
      memref.copy %4, %alloc_12 : memref<f32> to memref<f32>
      scf.for %arg6 = %c0 to %c1024 step %c1 {
        %22 = memref.load %3[%arg6] : memref<1024xf32>
        %23 = memref.load %alloc_12[] : memref<f32>
        %24 = arith.maximumf %22, %23 : f32
        memref.store %24, %alloc_12[] : memref<f32>
      }
      %5 = memref.load %alloc_12[] : memref<f32>
      memref.store %5, %alloc_11[%c0] : memref<1xf32>
      memref.copy %4, %alloc_13 : memref<f32> to memref<f32>
      %6 = memref.load %alloc_11[%c0] : memref<1xf32>
      %7 = memref.load %alloc_13[] : memref<f32>
      %8 = arith.maximumf %6, %7 : f32
      memref.store %8, %alloc_13[] : memref<f32>
      %9 = memref.load %alloc_13[] : memref<f32>
      %10 = upmem.alloc_dpus : !upmem.hierarchy<1x16x16>
      %11 = memref.get_global @__constant_2xi64_5 : memref<2xi64>
      %reshape = memref.reshape %3(%11) : (memref<1024xf32>, memref<2xi64>) -> memref<256x4xf32>
      upmem.scatter %reshape[0, 64, #map3] onto %10 : memref<256x4xf32> onto !upmem.hierarchy<1x16x16>
      memref.store %9, %alloc_14[%c0] : memref<16xf32>
      memref.store %9, %alloc_14[%c1] : memref<16xf32>
      memref.store %9, %alloc_14[%c2] : memref<16xf32>
      memref.store %9, %alloc_14[%c3] : memref<16xf32>
      memref.store %9, %alloc_14[%c4] : memref<16xf32>
      memref.store %9, %alloc_14[%c5] : memref<16xf32>
      memref.store %9, %alloc_14[%c6] : memref<16xf32>
      memref.store %9, %alloc_14[%c7] : memref<16xf32>
      memref.store %9, %alloc_14[%c8] : memref<16xf32>
      memref.store %9, %alloc_14[%c9] : memref<16xf32>
      memref.store %9, %alloc_14[%c10] : memref<16xf32>
      memref.store %9, %alloc_14[%c11] : memref<16xf32>
      memref.store %9, %alloc_14[%c12] : memref<16xf32>
      memref.store %9, %alloc_14[%c13] : memref<16xf32>
      memref.store %9, %alloc_14[%c14] : memref<16xf32>
      memref.store %9, %alloc_14[%c15] : memref<16xf32>
      upmem.scatter %alloc_14[256, 16, #map1] onto %10 : memref<16xf32> onto !upmem.hierarchy<1x16x16>
      %12 = memref.get_global @__constant_256x4xf32 : memref<256x4xf32>
      upmem.scatter %12[320, 64, #map3] onto %10 : memref<256x4xf32> onto !upmem.hierarchy<1x16x16>
      upmem.launch_func  @dpu_kernels::@mha_big_22 %10 : !upmem.hierarchy<1x16x16> 
      upmem.gather %alloc_15[320, 64, #map3] from %10 : memref<256x4xf32> from !upmem.hierarchy<1x16x16>
      %13 = memref.get_global @__constant_1xi64_5 : memref<1xi64>
      %reshape_27 = memref.reshape %alloc_15(%13) : (memref<256x4xf32>, memref<1xi64>) -> memref<1024xf32>
      upmem.free_dpus %10 : !upmem.hierarchy<1x16x16>
      scf.for %arg6 = %c0 to %c1024 step %c1 {
        %22 = memref.load %reshape_27[%arg6] : memref<1024xf32>
        %23 = llvm.intr.exp(%22)  : (f32) -> f32
        memref.store %23, %alloc_16[%arg6] : memref<1024xf32>
      }
      %14 = memref.get_global @__constant_xf32_0 : memref<f32>
      memref.copy %14, %alloc_18 : memref<f32> to memref<f32>
      scf.for %arg6 = %c0 to %c1024 step %c1 {
        %22 = memref.load %alloc_16[%arg6] : memref<1024xf32>
        %23 = memref.load %alloc_18[] : memref<f32>
        %24 = arith.addf %22, %23 : f32
        memref.store %24, %alloc_18[] : memref<f32>
      }
      %15 = memref.load %alloc_18[] : memref<f32>
      memref.store %15, %alloc_17[%c0] : memref<1xf32>
      memref.copy %14, %alloc_19 : memref<f32> to memref<f32>
      %16 = memref.load %alloc_17[%c0] : memref<1xf32>
      %17 = memref.load %alloc_19[] : memref<f32>
      %18 = arith.addf %16, %17 : f32
      memref.store %18, %alloc_19[] : memref<f32>
      %19 = memref.load %alloc_19[] : memref<f32>
      %20 = upmem.alloc_dpus : !upmem.hierarchy<1x16x16>
      %reshape_28 = memref.reshape %alloc_16(%11) : (memref<1024xf32>, memref<2xi64>) -> memref<256x4xf32>
      upmem.scatter %reshape_28[0, 64, #map3] onto %20 : memref<256x4xf32> onto !upmem.hierarchy<1x16x16>
      memref.store %19, %alloc_20[%c0] : memref<16xf32>
      memref.store %19, %alloc_20[%c1] : memref<16xf32>
      memref.store %19, %alloc_20[%c2] : memref<16xf32>
      memref.store %19, %alloc_20[%c3] : memref<16xf32>
      memref.store %19, %alloc_20[%c4] : memref<16xf32>
      memref.store %19, %alloc_20[%c5] : memref<16xf32>
      memref.store %19, %alloc_20[%c6] : memref<16xf32>
      memref.store %19, %alloc_20[%c7] : memref<16xf32>
      memref.store %19, %alloc_20[%c8] : memref<16xf32>
      memref.store %19, %alloc_20[%c9] : memref<16xf32>
      memref.store %19, %alloc_20[%c10] : memref<16xf32>
      memref.store %19, %alloc_20[%c11] : memref<16xf32>
      memref.store %19, %alloc_20[%c12] : memref<16xf32>
      memref.store %19, %alloc_20[%c13] : memref<16xf32>
      memref.store %19, %alloc_20[%c14] : memref<16xf32>
      memref.store %19, %alloc_20[%c15] : memref<16xf32>
      upmem.scatter %alloc_20[256, 16, #map1] onto %20 : memref<16xf32> onto !upmem.hierarchy<1x16x16>
      upmem.scatter %12[320, 64, #map3] onto %20 : memref<256x4xf32> onto !upmem.hierarchy<1x16x16>
      upmem.launch_func  @dpu_kernels::@mha_big_23 %20 : !upmem.hierarchy<1x16x16> 
      upmem.gather %alloc_21[320, 64, #map3] from %20 : memref<256x4xf32> from !upmem.hierarchy<1x16x16>
      %reshape_29 = memref.reshape %alloc_21(%13) : (memref<256x4xf32>, memref<1xi64>) -> memref<1024xf32>
      upmem.free_dpus %20 : !upmem.hierarchy<1x16x16>
      scf.for %arg6 = %c0 to %c4096 step %c1 {
        memref.store %cst, %alloc_22[%arg6] : memref<4096xf32>
      }
      memref.copy %alloc_22, %alloc_23 : memref<4096xf32> to memref<4096xf32>
      %21 = scf.for %arg6 = %c0 to %0 step %c1 iter_args(%arg7 = %alloc_23) -> (memref<4096xf32>) {
        %subview_31 = memref.subview %arg2[%arg6, %2] [1, 4096] [1, 1] : memref<1024x32768xf32> to memref<4096xf32, strided<[1], offset: ?>>
        %22 = memref.load %reshape_29[%arg6] : memref<1024xf32>
        %23 = upmem.alloc_dpus : !upmem.hierarchy<1x16x16>
        %24 = memref.get_global @__constant_2xi64_4 : memref<2xi64>
        memref.copy %subview_31, %alloc_24 : memref<4096xf32, strided<[1], offset: ?>> to memref<4096xf32>
        %reshape_32 = memref.reshape %alloc_24(%24) : (memref<4096xf32>, memref<2xi64>) -> memref<256x16xf32>
        upmem.scatter %reshape_32[0, 256, #map3] onto %23 : memref<256x16xf32> onto !upmem.hierarchy<1x16x16>
        memref.store %22, %alloc_25[%c0] : memref<16xf32>
        memref.store %22, %alloc_25[%c1] : memref<16xf32>
        memref.store %22, %alloc_25[%c2] : memref<16xf32>
        memref.store %22, %alloc_25[%c3] : memref<16xf32>
        memref.store %22, %alloc_25[%c4] : memref<16xf32>
        memref.store %22, %alloc_25[%c5] : memref<16xf32>
        memref.store %22, %alloc_25[%c6] : memref<16xf32>
        memref.store %22, %alloc_25[%c7] : memref<16xf32>
        memref.store %22, %alloc_25[%c8] : memref<16xf32>
        memref.store %22, %alloc_25[%c9] : memref<16xf32>
        memref.store %22, %alloc_25[%c10] : memref<16xf32>
        memref.store %22, %alloc_25[%c11] : memref<16xf32>
        memref.store %22, %alloc_25[%c12] : memref<16xf32>
        memref.store %22, %alloc_25[%c13] : memref<16xf32>
        memref.store %22, %alloc_25[%c14] : memref<16xf32>
        memref.store %22, %alloc_25[%c15] : memref<16xf32>
        upmem.scatter %alloc_25[1024, 16, #map1] onto %23 : memref<16xf32> onto !upmem.hierarchy<1x16x16>
        %25 = memref.get_global @__constant_256x16xf32 : memref<256x16xf32>
        upmem.scatter %25[1088, 256, #map3] onto %23 : memref<256x16xf32> onto !upmem.hierarchy<1x16x16>
        upmem.launch_func  @dpu_kernels::@mha_big_24 %23 : !upmem.hierarchy<1x16x16> 
        upmem.gather %alloc_26[1088, 256, #map3] from %23 : memref<256x16xf32> from !upmem.hierarchy<1x16x16>
        %26 = memref.get_global @__constant_1xi64_4 : memref<1xi64>
        %reshape_33 = memref.reshape %alloc_26(%26) : (memref<256x16xf32>, memref<1xi64>) -> memref<4096xf32>
        upmem.free_dpus %23 : !upmem.hierarchy<1x16x16>
        %27 = upmem.alloc_dpus : !upmem.hierarchy<1x16x16>
        %reshape_34 = memref.reshape %arg7(%24) : (memref<4096xf32>, memref<2xi64>) -> memref<256x16xf32>
        upmem.scatter %reshape_34[0, 256, #map3] onto %27 : memref<256x16xf32> onto !upmem.hierarchy<1x16x16>
        %reshape_35 = memref.reshape %reshape_33(%24) : (memref<4096xf32>, memref<2xi64>) -> memref<256x16xf32>
        upmem.scatter %reshape_35[1024, 256, #map3] onto %27 : memref<256x16xf32> onto !upmem.hierarchy<1x16x16>
        upmem.scatter %25[2048, 256, #map3] onto %27 : memref<256x16xf32> onto !upmem.hierarchy<1x16x16>
        upmem.launch_func  @dpu_kernels::@mha_big_25 %27 : !upmem.hierarchy<1x16x16> 
        %alloc_36 = memref.alloc() : memref<256x16xf32>
        upmem.gather %alloc_36[2048, 256, #map3] from %27 : memref<256x16xf32> from !upmem.hierarchy<1x16x16>
        %reshape_37 = memref.reshape %alloc_36(%26) : (memref<256x16xf32>, memref<1xi64>) -> memref<4096xf32>
        upmem.free_dpus %27 : !upmem.hierarchy<1x16x16>
        scf.yield %reshape_37 : memref<4096xf32>
      }
      %alloc_30 = memref.alloc() : memref<32768xf32>
      memref.copy %arg5, %alloc_30 : memref<32768xf32> to memref<32768xf32>
      %subview = memref.subview %alloc_30[%2] [4096] [1] : memref<32768xf32> to memref<4096xf32, strided<[1], offset: ?>>
      memref.copy %21, %subview : memref<4096xf32> to memref<4096xf32, strided<[1], offset: ?>>
      scf.yield %alloc_30 : memref<32768xf32>
    }
    return %1 : memref<32768xf32>
  }
  func.func @test_0(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>, %arg2: memref<1024xi32>, %arg3: memref<1024xi32>) {
    %0 = upmem.alloc_dpus : !upmem.hierarchy<1x16x16>
    %1 = upmem.alloc_dpus : !upmem.hierarchy<1x16x16>
    %2 = upmem.alloc_dpus : !upmem.hierarchy<1x16x16>
    %3 = memref.get_global @__tconstant_2xi64 : memref<2xi64>
    %reshape = memref.reshape %arg0(%3) : (memref<1024xi32>, memref<2xi64>) -> memref<256x4xi32>
    upmem.scatter %reshape[0, 64, #map3] onto %0 : memref<256x4xi32> onto !upmem.hierarchy<1x16x16>
    %reshape_0 = memref.reshape %arg1(%3) : (memref<1024xi32>, memref<2xi64>) -> memref<256x4xi32>
    upmem.scatter %reshape_0[256, 64, #map3] onto %0 : memref<256x4xi32> onto !upmem.hierarchy<1x16x16>
    %4 = memref.get_global @__tconstant_256x4xi32 : memref<256x4xi32>
    upmem.scatter %4[512, 64, #map3] onto %0 : memref<256x4xi32> onto !upmem.hierarchy<1x16x16>
    upmem.scatter %reshape[0, 64, #map3] onto %1 : memref<256x4xi32> onto !upmem.hierarchy<1x16x16>
    %reshape_1 = memref.reshape %arg2(%3) : (memref<1024xi32>, memref<2xi64>) -> memref<256x4xi32>
    upmem.scatter %reshape_1[256, 64, #map3] onto %1 : memref<256x4xi32> onto !upmem.hierarchy<1x16x16>
    upmem.scatter %4[512, 64, #map3] onto %1 : memref<256x4xi32> onto !upmem.hierarchy<1x16x16>
    upmem.scatter %reshape[0, 64, #map3] onto %2 : memref<256x4xi32> onto !upmem.hierarchy<1x16x16>
    %reshape_2 = memref.reshape %arg3(%3) : (memref<1024xi32>, memref<2xi64>) -> memref<256x4xi32>
    upmem.scatter %reshape_2[256, 64, #map3] onto %2 : memref<256x4xi32> onto !upmem.hierarchy<1x16x16>
    upmem.scatter %4[512, 64, #map3] onto %2 : memref<256x4xi32> onto !upmem.hierarchy<1x16x16>
    upmem.launch_func  @dpu_kernels::@test_0 %0 : !upmem.hierarchy<1x16x16> 
    upmem.launch_func  @dpu_kernels::@test_0 %1 : !upmem.hierarchy<1x16x16> 
    upmem.launch_func  @dpu_kernels::@test_0 %2 : !upmem.hierarchy<1x16x16> 
    %alloc = memref.alloc() : memref<256x4xi32>
    upmem.gather %alloc[512, 64, #map3] from %0 : memref<256x4xi32> from !upmem.hierarchy<1x16x16>
    %alloc_3 = memref.alloc() : memref<256x4xi32>
    upmem.gather %alloc_3[512, 64, #map3] from %1 : memref<256x4xi32> from !upmem.hierarchy<1x16x16>
    %alloc_4 = memref.alloc() : memref<256x4xi32>
    upmem.gather %alloc_4[512, 64, #map3] from %2 : memref<256x4xi32> from !upmem.hierarchy<1x16x16>
    upmem.free_dpus %0 : !upmem.hierarchy<1x16x16>
    upmem.free_dpus %1 : !upmem.hierarchy<1x16x16>
    upmem.free_dpus %2 : !upmem.hierarchy<1x16x16>
    return
  }
  func.func @test_1(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>, %arg2: memref<1024xi32>, %arg3: memref<1024xi32>) {
    %0 = upmem.alloc_dpus : !upmem.hierarchy<1x16x16>
    %1 = memref.get_global @__tconstant_2xi64 : memref<2xi64>
    %reshape = memref.reshape %arg0(%1) : (memref<1024xi32>, memref<2xi64>) -> memref<256x4xi32>
    upmem.scatter %reshape[0, 64, #map3] onto %0 : memref<256x4xi32> onto !upmem.hierarchy<1x16x16>
    %reshape_0 = memref.reshape %arg1(%1) : (memref<1024xi32>, memref<2xi64>) -> memref<256x4xi32>
    upmem.scatter %reshape_0[256, 64, #map3] onto %0 : memref<256x4xi32> onto !upmem.hierarchy<1x16x16>
    %2 = memref.get_global @__tconstant_256x4xi32 : memref<256x4xi32>
    upmem.scatter %2[512, 64, #map3] onto %0 : memref<256x4xi32> onto !upmem.hierarchy<1x16x16>
    upmem.scatter %reshape[768, 64, #map3] onto %0 : memref<256x4xi32> onto !upmem.hierarchy<1x16x16>
    %reshape_1 = memref.reshape %arg2(%1) : (memref<1024xi32>, memref<2xi64>) -> memref<256x4xi32>
    upmem.scatter %reshape_1[1024, 64, #map3] onto %0 : memref<256x4xi32> onto !upmem.hierarchy<1x16x16>
    upmem.scatter %2[1280, 64, #map3] onto %0 : memref<256x4xi32> onto !upmem.hierarchy<1x16x16>
    upmem.scatter %reshape[1536, 64, #map3] onto %0 : memref<256x4xi32> onto !upmem.hierarchy<1x16x16>
    %reshape_2 = memref.reshape %arg3(%1) : (memref<1024xi32>, memref<2xi64>) -> memref<256x4xi32>
    upmem.scatter %reshape_2[1792, 64, #map3] onto %0 : memref<256x4xi32> onto !upmem.hierarchy<1x16x16>
    upmem.scatter %2[2048, 64, #map3] onto %0 : memref<256x4xi32> onto !upmem.hierarchy<1x16x16>
    upmem.launch_func  @dpu_kernels::@test_0 %0 : !upmem.hierarchy<1x16x16> 
    upmem.launch_func  @dpu_kernels::@test_0 %0 : !upmem.hierarchy<1x16x16> 
    upmem.launch_func  @dpu_kernels::@test_0 %0 : !upmem.hierarchy<1x16x16> 
    %alloc = memref.alloc() : memref<256x4xi32>
    upmem.gather %alloc[512, 64, #map3] from %0 : memref<256x4xi32> from !upmem.hierarchy<1x16x16>
    %alloc_3 = memref.alloc() : memref<256x4xi32>
    upmem.gather %alloc_3[1280, 64, #map3] from %0 : memref<256x4xi32> from !upmem.hierarchy<1x16x16>
    %alloc_4 = memref.alloc() : memref<256x4xi32>
    upmem.gather %alloc_4[2048, 64, #map3] from %0 : memref<256x4xi32> from !upmem.hierarchy<1x16x16>
    upmem.free_dpus %0 : !upmem.hierarchy<1x16x16>
    return
  }
  func.func @test_2(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>, %arg2: memref<1024xi32>, %arg3: memref<1024xi32>) {
    %0 = upmem.alloc_dpus : !upmem.hierarchy<1x16x16>
    %1 = memref.get_global @__tconstant_2xi64 : memref<2xi64>
    %reshape = memref.reshape %arg0(%1) : (memref<1024xi32>, memref<2xi64>) -> memref<256x4xi32>
    upmem.scatter %reshape[0, 64, #map3] onto %0 : memref<256x4xi32> onto !upmem.hierarchy<1x16x16>
    %reshape_0 = memref.reshape %arg1(%1) : (memref<1024xi32>, memref<2xi64>) -> memref<256x4xi32>
    upmem.scatter %reshape_0[256, 64, #map3] onto %0 : memref<256x4xi32> onto !upmem.hierarchy<1x16x16>
    %2 = memref.get_global @__tconstant_256x4xi32 : memref<256x4xi32>
    upmem.scatter %2[512, 64, #map3] onto %0 : memref<256x4xi32> onto !upmem.hierarchy<1x16x16>
    upmem.scatter %reshape[768, 64, #map3] onto %0 : memref<256x4xi32> onto !upmem.hierarchy<1x16x16>
    %reshape_1 = memref.reshape %arg2(%1) : (memref<1024xi32>, memref<2xi64>) -> memref<256x4xi32>
    upmem.scatter %reshape_1[1024, 64, #map3] onto %0 : memref<256x4xi32> onto !upmem.hierarchy<1x16x16>
    upmem.scatter %2[1280, 64, #map3] onto %0 : memref<256x4xi32> onto !upmem.hierarchy<1x16x16>
    upmem.scatter %reshape[1536, 64, #map3] onto %0 : memref<256x4xi32> onto !upmem.hierarchy<1x16x16>
    %reshape_2 = memref.reshape %arg3(%1) : (memref<1024xi32>, memref<2xi64>) -> memref<256x4xi32>
    upmem.scatter %reshape_2[1792, 64, #map3] onto %0 : memref<256x4xi32> onto !upmem.hierarchy<1x16x16>
    upmem.scatter %2[2048, 64, #map3] onto %0 : memref<256x4xi32> onto !upmem.hierarchy<1x16x16>
    upmem.launch_func  @dpu_kernels::@test_2 %0 : !upmem.hierarchy<1x16x16> 
    %alloc = memref.alloc() : memref<256x4xi32>
    upmem.gather %alloc[512, 64, #map3] from %0 : memref<256x4xi32> from !upmem.hierarchy<1x16x16>
    %alloc_3 = memref.alloc() : memref<256x4xi32>
    upmem.gather %alloc_3[1280, 64, #map3] from %0 : memref<256x4xi32> from !upmem.hierarchy<1x16x16>
    %alloc_4 = memref.alloc() : memref<256x4xi32>
    upmem.gather %alloc_4[2048, 64, #map3] from %0 : memref<256x4xi32> from !upmem.hierarchy<1x16x16>
    upmem.free_dpus %0 : !upmem.hierarchy<1x16x16>
    return
  }
  upmem.module @dpu_kernels {
    upmem.func @forward() attributes {num_tasklets = 8 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c1152 = arith.constant 1152 : index
      %2 = arith.muli %1, %c1152 : index
      %3 = arith.addi %0, %2 : index
      %c288 = arith.constant 288 : index
      %4 = upmem.pwram_alloc : memref<288xf32>
      %c9216 = arith.constant 9216 : index
      %5 = arith.addi %0, %c9216 : index
      %6 = arith.addi %5, %2 : index
      %7 = upmem.pwram_alloc : memref<288xf32>
      %8 = arith.addi %5, %c9216 : index
      %c8 = arith.constant 8 : index
      %9 = arith.muli %1, %c8 : index
      %10 = arith.addi %8, %9 : index
      %c2 = arith.constant 2 : index
      %11 = upmem.pwram_alloc : memref<f32>
      upmem.memcpy  mram_to_wram %4, %c288, %3 : memref<288xf32>, index, index
      upmem.memcpy  mram_to_wram %7, %c288, %6 : memref<288xf32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c288 step %c1 {
        %12 = memref.load %4[%arg0] : memref<288xf32>
        %13 = memref.load %7[%arg0] : memref<288xf32>
        %14 = memref.load %11[] : memref<f32>
        %15 = arith.mulf %12, %13 : f32
        %16 = arith.addf %15, %14 : f32
        memref.store %16, %11[] : memref<f32>
      }
      upmem.memcpy  wram_to_mram %11, %c2, %10 : memref<f32>, index, index
      upmem.return
    }
    upmem.func @forward_3() attributes {num_tasklets = 8 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c24 = arith.constant 24 : index
      %2 = arith.muli %1, %c24 : index
      %3 = arith.addi %0, %2 : index
      %c6 = arith.constant 6 : index
      %4 = upmem.pwram_alloc : memref<6xf32>
      %c192 = arith.constant 192 : index
      %5 = arith.addi %0, %c192 : index
      %6 = arith.addi %5, %2 : index
      %7 = upmem.pwram_alloc : memref<6xf32>
      %8 = arith.addi %5, %c192 : index
      %9 = arith.addi %8, %2 : index
      %10 = upmem.pwram_alloc : memref<6xf32>
      upmem.memcpy  mram_to_wram %4, %c6, %3 : memref<6xf32>, index, index
      upmem.memcpy  mram_to_wram %7, %c6, %6 : memref<6xf32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c6 step %c1 {
        %11 = memref.load %4[%arg0] : memref<6xf32>
        %12 = memref.load %7[%arg0] : memref<6xf32>
        %13 = arith.addf %11, %12 : f32
        memref.store %13, %10[%arg0] : memref<6xf32>
      }
      upmem.memcpy  wram_to_mram %10, %c6, %9 : memref<6xf32>, index, index
      upmem.return
    }
    upmem.func @forward_6() attributes {num_tasklets = 8 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c3072 = arith.constant 3072 : index
      %2 = arith.muli %1, %c3072 : index
      %3 = arith.addi %0, %2 : index
      %c768 = arith.constant 768 : index
      %4 = upmem.pwram_alloc : memref<768xf32>
      %c24576 = arith.constant 24576 : index
      %5 = arith.addi %0, %c24576 : index
      %6 = arith.addi %5, %2 : index
      %7 = upmem.pwram_alloc : memref<768xf32>
      %8 = arith.addi %5, %c24576 : index
      %c8 = arith.constant 8 : index
      %9 = arith.muli %1, %c8 : index
      %10 = arith.addi %8, %9 : index
      %c2 = arith.constant 2 : index
      %11 = upmem.pwram_alloc : memref<f32>
      upmem.memcpy  mram_to_wram %4, %c768, %3 : memref<768xf32>, index, index
      upmem.memcpy  mram_to_wram %7, %c768, %6 : memref<768xf32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c768 step %c1 {
        %12 = memref.load %4[%arg0] : memref<768xf32>
        %13 = memref.load %7[%arg0] : memref<768xf32>
        %14 = memref.load %11[] : memref<f32>
        %15 = arith.mulf %12, %13 : f32
        %16 = arith.addf %15, %14 : f32
        memref.store %16, %11[] : memref<f32>
      }
      upmem.memcpy  wram_to_mram %11, %c2, %10 : memref<f32>, index, index
      upmem.return
    }
    upmem.func @forward_8() attributes {num_tasklets = 16 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c1152 = arith.constant 1152 : index
      %2 = arith.muli %1, %c1152 : index
      %3 = arith.addi %0, %2 : index
      %c288 = arith.constant 288 : index
      %4 = upmem.pwram_alloc : memref<288xf32>
      %c18432 = arith.constant 18432 : index
      %5 = arith.addi %0, %c18432 : index
      %6 = arith.addi %5, %2 : index
      %7 = upmem.pwram_alloc : memref<288xf32>
      %8 = arith.addi %5, %c18432 : index
      %c8 = arith.constant 8 : index
      %9 = arith.muli %1, %c8 : index
      %10 = arith.addi %8, %9 : index
      %c2 = arith.constant 2 : index
      %11 = upmem.pwram_alloc : memref<f32>
      upmem.memcpy  mram_to_wram %4, %c288, %3 : memref<288xf32>, index, index
      upmem.memcpy  mram_to_wram %7, %c288, %6 : memref<288xf32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c288 step %c1 {
        %12 = memref.load %4[%arg0] : memref<288xf32>
        %13 = memref.load %7[%arg0] : memref<288xf32>
        %14 = memref.load %11[] : memref<f32>
        %15 = arith.mulf %12, %13 : f32
        %16 = arith.addf %15, %14 : f32
        memref.store %16, %11[] : memref<f32>
      }
      upmem.memcpy  wram_to_mram %11, %c2, %10 : memref<f32>, index, index
      upmem.return
    }
    upmem.func @mha() attributes {num_tasklets = 8 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c24 = arith.constant 24 : index
      %2 = arith.muli %1, %c24 : index
      %3 = arith.addi %0, %2 : index
      %c6 = arith.constant 6 : index
      %4 = upmem.pwram_alloc : memref<6xf32>
      %c192 = arith.constant 192 : index
      %5 = arith.addi %0, %c192 : index
      %6 = arith.addi %5, %2 : index
      %7 = upmem.pwram_alloc : memref<6xf32>
      %8 = arith.addi %5, %c192 : index
      %9 = arith.addi %8, %2 : index
      %10 = upmem.pwram_alloc : memref<6xf32>
      upmem.memcpy  mram_to_wram %4, %c6, %3 : memref<6xf32>, index, index
      upmem.memcpy  mram_to_wram %7, %c6, %6 : memref<6xf32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c6 step %c1 {
        %11 = memref.load %4[%arg0] : memref<6xf32>
        %12 = memref.load %7[%arg0] : memref<6xf32>
        %13 = arith.mulf %11, %12 : f32
        memref.store %13, %10[%arg0] : memref<6xf32>
      }
      upmem.memcpy  wram_to_mram %10, %c6, %9 : memref<6xf32>, index, index
      upmem.return
    }
    upmem.func @mha_9() attributes {num_tasklets = 8 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c24 = arith.constant 24 : index
      %2 = arith.muli %1, %c24 : index
      %3 = arith.addi %0, %2 : index
      %c6 = arith.constant 6 : index
      %4 = upmem.pwram_alloc : memref<6xf32>
      %c192 = arith.constant 192 : index
      %5 = arith.addi %0, %c192 : index
      %c2 = arith.constant 2 : index
      %6 = upmem.pwram_alloc : memref<f32>
      %c32 = arith.constant 32 : index
      %7 = arith.addi %5, %c32 : index
      %8 = arith.addi %7, %2 : index
      %9 = upmem.pwram_alloc : memref<6xf32>
      upmem.memcpy  mram_to_wram %4, %c6, %3 : memref<6xf32>, index, index
      upmem.memcpy  mram_to_wram %6, %c2, %5 : memref<f32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c6 step %c1 {
        %10 = memref.load %4[%arg0] : memref<6xf32>
        %11 = memref.load %6[] : memref<f32>
        %12 = arith.mulf %10, %11 : f32
        memref.store %12, %9[%arg0] : memref<6xf32>
      }
      upmem.memcpy  wram_to_mram %9, %c6, %8 : memref<6xf32>, index, index
      upmem.return
    }
    upmem.func @rmsnorm() attributes {num_tasklets = 16 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c72 = arith.constant 72 : index
      %2 = arith.muli %1, %c72 : index
      %3 = arith.addi %0, %2 : index
      %c18 = arith.constant 18 : index
      %4 = upmem.pwram_alloc : memref<18xf32>
      %c1152 = arith.constant 1152 : index
      %5 = arith.addi %0, %c1152 : index
      %6 = arith.addi %5, %2 : index
      %7 = upmem.pwram_alloc : memref<18xf32>
      %8 = arith.addi %5, %c1152 : index
      %9 = arith.addi %8, %2 : index
      %10 = upmem.pwram_alloc : memref<18xf32>
      upmem.memcpy  mram_to_wram %4, %c18, %3 : memref<18xf32>, index, index
      upmem.memcpy  mram_to_wram %7, %c18, %6 : memref<18xf32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c18 step %c1 {
        %11 = memref.load %4[%arg0] : memref<18xf32>
        %12 = memref.load %7[%arg0] : memref<18xf32>
        %13 = arith.mulf %11, %12 : f32
        memref.store %13, %10[%arg0] : memref<18xf32>
      }
      upmem.memcpy  wram_to_mram %10, %c18, %9 : memref<18xf32>, index, index
      upmem.return
    }
    upmem.func @rmsnorm_11() attributes {num_tasklets = 16 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c72 = arith.constant 72 : index
      %2 = arith.muli %1, %c72 : index
      %3 = arith.addi %0, %2 : index
      %c18 = arith.constant 18 : index
      %4 = upmem.pwram_alloc : memref<18xf32>
      %c1152 = arith.constant 1152 : index
      %5 = arith.addi %0, %c1152 : index
      %c2 = arith.constant 2 : index
      %6 = upmem.pwram_alloc : memref<f32>
      %c64 = arith.constant 64 : index
      %7 = arith.addi %5, %c64 : index
      %8 = arith.addi %7, %2 : index
      %9 = upmem.pwram_alloc : memref<18xf32>
      upmem.memcpy  mram_to_wram %4, %c18, %3 : memref<18xf32>, index, index
      upmem.memcpy  mram_to_wram %6, %c2, %5 : memref<f32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c18 step %c1 {
        %10 = memref.load %4[%arg0] : memref<18xf32>
        %11 = memref.load %6[] : memref<f32>
        %12 = arith.mulf %10, %11 : f32
        memref.store %12, %9[%arg0] : memref<18xf32>
      }
      upmem.memcpy  wram_to_mram %9, %c18, %8 : memref<18xf32>, index, index
      upmem.return
    }
    upmem.func @softmax() attributes {num_tasklets = 16 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c8 = arith.constant 8 : index
      %2 = arith.muli %1, %c8 : index
      %3 = arith.addi %0, %2 : index
      %c2 = arith.constant 2 : index
      %4 = upmem.pwram_alloc : memref<2xf32>
      %c128 = arith.constant 128 : index
      %5 = arith.addi %0, %c128 : index
      %6 = upmem.pwram_alloc : memref<f32>
      %c64 = arith.constant 64 : index
      %7 = arith.addi %5, %c64 : index
      %8 = arith.addi %7, %2 : index
      %9 = upmem.pwram_alloc : memref<2xf32>
      upmem.memcpy  mram_to_wram %4, %c2, %3 : memref<2xf32>, index, index
      upmem.memcpy  mram_to_wram %6, %c2, %5 : memref<f32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %10 = memref.load %4[%arg0] : memref<2xf32>
        %11 = memref.load %6[] : memref<f32>
        %12 = arith.subf %10, %11 : f32
        memref.store %12, %9[%arg0] : memref<2xf32>
      }
      upmem.memcpy  wram_to_mram %9, %c2, %8 : memref<2xf32>, index, index
      upmem.return
    }
    upmem.func @softmax_13() attributes {num_tasklets = 16 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c8 = arith.constant 8 : index
      %2 = arith.muli %1, %c8 : index
      %3 = arith.addi %0, %2 : index
      %c2 = arith.constant 2 : index
      %4 = upmem.pwram_alloc : memref<2xf32>
      %c128 = arith.constant 128 : index
      %5 = arith.addi %0, %c128 : index
      %6 = upmem.pwram_alloc : memref<f32>
      %c64 = arith.constant 64 : index
      %7 = arith.addi %5, %c64 : index
      %8 = arith.addi %7, %2 : index
      %9 = upmem.pwram_alloc : memref<2xf32>
      upmem.memcpy  mram_to_wram %4, %c2, %3 : memref<2xf32>, index, index
      upmem.memcpy  mram_to_wram %6, %c2, %5 : memref<f32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %10 = memref.load %4[%arg0] : memref<2xf32>
        %11 = memref.load %6[] : memref<f32>
        %12 = arith.divf %10, %11 : f32
        memref.store %12, %9[%arg0] : memref<2xf32>
      }
      upmem.memcpy  wram_to_mram %9, %c2, %8 : memref<2xf32>, index, index
      upmem.return
    }
    upmem.func @rmsnorm_1048576() attributes {num_tasklets = 16 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c1024 = arith.constant 1024 : index
      %2 = arith.muli %1, %c1024 : index
      %3 = arith.addi %0, %2 : index
      %c256 = arith.constant 256 : index
      %4 = upmem.pwram_alloc : memref<256xf32>
      %c16384 = arith.constant 16384 : index
      %5 = arith.addi %0, %c16384 : index
      %6 = arith.addi %5, %2 : index
      %7 = upmem.pwram_alloc : memref<256xf32>
      %8 = arith.addi %5, %c16384 : index
      %9 = arith.addi %8, %2 : index
      %10 = upmem.pwram_alloc : memref<256xf32>
      upmem.memcpy  mram_to_wram %4, %c256, %3 : memref<256xf32>, index, index
      upmem.memcpy  mram_to_wram %7, %c256, %6 : memref<256xf32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c256 step %c1 {
        %11 = memref.load %4[%arg0] : memref<256xf32>
        %12 = memref.load %7[%arg0] : memref<256xf32>
        %13 = arith.mulf %11, %12 : f32
        memref.store %13, %10[%arg0] : memref<256xf32>
      }
      upmem.memcpy  wram_to_mram %10, %c256, %9 : memref<256xf32>, index, index
      upmem.return
    }
    upmem.func @rmsnorm_1048576_14() attributes {num_tasklets = 16 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c1024 = arith.constant 1024 : index
      %2 = arith.muli %1, %c1024 : index
      %3 = arith.addi %0, %2 : index
      %c256 = arith.constant 256 : index
      %4 = upmem.pwram_alloc : memref<256xf32>
      %c16384 = arith.constant 16384 : index
      %5 = arith.addi %0, %c16384 : index
      %c2 = arith.constant 2 : index
      %6 = upmem.pwram_alloc : memref<f32>
      %c64 = arith.constant 64 : index
      %7 = arith.addi %5, %c64 : index
      %8 = arith.addi %7, %2 : index
      %9 = upmem.pwram_alloc : memref<256xf32>
      upmem.memcpy  mram_to_wram %4, %c256, %3 : memref<256xf32>, index, index
      upmem.memcpy  mram_to_wram %6, %c2, %5 : memref<f32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c256 step %c1 {
        %10 = memref.load %4[%arg0] : memref<256xf32>
        %11 = memref.load %6[] : memref<f32>
        %12 = arith.mulf %10, %11 : f32
        memref.store %12, %9[%arg0] : memref<256xf32>
      }
      upmem.memcpy  wram_to_mram %9, %c256, %8 : memref<256xf32>, index, index
      upmem.return
    }
    upmem.func @softmax_1048576() attributes {num_tasklets = 16 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c1024 = arith.constant 1024 : index
      %2 = arith.muli %1, %c1024 : index
      %3 = arith.addi %0, %2 : index
      %c256 = arith.constant 256 : index
      %4 = upmem.pwram_alloc : memref<256xf32>
      %c16384 = arith.constant 16384 : index
      %5 = arith.addi %0, %c16384 : index
      %c2 = arith.constant 2 : index
      %6 = upmem.pwram_alloc : memref<f32>
      %c64 = arith.constant 64 : index
      %7 = arith.addi %5, %c64 : index
      %8 = arith.addi %7, %2 : index
      %9 = upmem.pwram_alloc : memref<256xf32>
      upmem.memcpy  mram_to_wram %4, %c256, %3 : memref<256xf32>, index, index
      upmem.memcpy  mram_to_wram %6, %c2, %5 : memref<f32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c256 step %c1 {
        %10 = memref.load %4[%arg0] : memref<256xf32>
        %11 = memref.load %6[] : memref<f32>
        %12 = arith.subf %10, %11 : f32
        memref.store %12, %9[%arg0] : memref<256xf32>
      }
      upmem.memcpy  wram_to_mram %9, %c256, %8 : memref<256xf32>, index, index
      upmem.return
    }
    upmem.func @softmax_1048576_16() attributes {num_tasklets = 16 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c1024 = arith.constant 1024 : index
      %2 = arith.muli %1, %c1024 : index
      %3 = arith.addi %0, %2 : index
      %c256 = arith.constant 256 : index
      %4 = upmem.pwram_alloc : memref<256xf32>
      %c16384 = arith.constant 16384 : index
      %5 = arith.addi %0, %c16384 : index
      %c2 = arith.constant 2 : index
      %6 = upmem.pwram_alloc : memref<f32>
      %c64 = arith.constant 64 : index
      %7 = arith.addi %5, %c64 : index
      %8 = arith.addi %7, %2 : index
      %9 = upmem.pwram_alloc : memref<256xf32>
      upmem.memcpy  mram_to_wram %4, %c256, %3 : memref<256xf32>, index, index
      upmem.memcpy  mram_to_wram %6, %c2, %5 : memref<f32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c256 step %c1 {
        %10 = memref.load %4[%arg0] : memref<256xf32>
        %11 = memref.load %6[] : memref<f32>
        %12 = arith.divf %10, %11 : f32
        memref.store %12, %9[%arg0] : memref<256xf32>
      }
      upmem.memcpy  wram_to_mram %9, %c256, %8 : memref<256xf32>, index, index
      upmem.return
    }
    upmem.func @va_1048576() attributes {num_tasklets = 16 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c1024 = arith.constant 1024 : index
      %2 = arith.muli %1, %c1024 : index
      %3 = arith.addi %0, %2 : index
      %c256 = arith.constant 256 : index
      %4 = upmem.pwram_alloc : memref<256xf32>
      %c16384 = arith.constant 16384 : index
      %5 = arith.addi %0, %c16384 : index
      %6 = arith.addi %5, %2 : index
      %7 = upmem.pwram_alloc : memref<256xf32>
      %8 = arith.addi %5, %c16384 : index
      %9 = arith.addi %8, %2 : index
      %10 = upmem.pwram_alloc : memref<256xf32>
      upmem.memcpy  mram_to_wram %4, %c256, %3 : memref<256xf32>, index, index
      upmem.memcpy  mram_to_wram %7, %c256, %6 : memref<256xf32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c256 step %c1 {
        %11 = memref.load %4[%arg0] : memref<256xf32>
        %12 = memref.load %7[%arg0] : memref<256xf32>
        %13 = arith.addf %11, %12 : f32
        memref.store %13, %10[%arg0] : memref<256xf32>
      }
      upmem.memcpy  wram_to_mram %10, %c256, %9 : memref<256xf32>, index, index
      upmem.return
    }
    upmem.func @rmsnorm_262144() attributes {num_tasklets = 16 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c256 = arith.constant 256 : index
      %2 = arith.muli %1, %c256 : index
      %3 = arith.addi %0, %2 : index
      %c64 = arith.constant 64 : index
      %4 = upmem.pwram_alloc : memref<64xf32>
      %c4096 = arith.constant 4096 : index
      %5 = arith.addi %0, %c4096 : index
      %6 = arith.addi %5, %2 : index
      %7 = upmem.pwram_alloc : memref<64xf32>
      %8 = arith.addi %5, %c4096 : index
      %9 = arith.addi %8, %2 : index
      %10 = upmem.pwram_alloc : memref<64xf32>
      upmem.memcpy  mram_to_wram %4, %c64, %3 : memref<64xf32>, index, index
      upmem.memcpy  mram_to_wram %7, %c64, %6 : memref<64xf32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c64 step %c1 {
        %11 = memref.load %4[%arg0] : memref<64xf32>
        %12 = memref.load %7[%arg0] : memref<64xf32>
        %13 = arith.mulf %11, %12 : f32
        memref.store %13, %10[%arg0] : memref<64xf32>
      }
      upmem.memcpy  wram_to_mram %10, %c64, %9 : memref<64xf32>, index, index
      upmem.return
    }
    upmem.func @rmsnorm_262144_17() attributes {num_tasklets = 16 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c256 = arith.constant 256 : index
      %2 = arith.muli %1, %c256 : index
      %3 = arith.addi %0, %2 : index
      %c64 = arith.constant 64 : index
      %4 = upmem.pwram_alloc : memref<64xf32>
      %c4096 = arith.constant 4096 : index
      %5 = arith.addi %0, %c4096 : index
      %c2 = arith.constant 2 : index
      %6 = upmem.pwram_alloc : memref<f32>
      %7 = arith.addi %5, %c64 : index
      %8 = arith.addi %7, %2 : index
      %9 = upmem.pwram_alloc : memref<64xf32>
      upmem.memcpy  mram_to_wram %4, %c64, %3 : memref<64xf32>, index, index
      upmem.memcpy  mram_to_wram %6, %c2, %5 : memref<f32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c64 step %c1 {
        %10 = memref.load %4[%arg0] : memref<64xf32>
        %11 = memref.load %6[] : memref<f32>
        %12 = arith.mulf %10, %11 : f32
        memref.store %12, %9[%arg0] : memref<64xf32>
      }
      upmem.memcpy  wram_to_mram %9, %c64, %8 : memref<64xf32>, index, index
      upmem.return
    }
    upmem.func @softmax_262144() attributes {num_tasklets = 16 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c256 = arith.constant 256 : index
      %2 = arith.muli %1, %c256 : index
      %3 = arith.addi %0, %2 : index
      %c64 = arith.constant 64 : index
      %4 = upmem.pwram_alloc : memref<64xf32>
      %c4096 = arith.constant 4096 : index
      %5 = arith.addi %0, %c4096 : index
      %c2 = arith.constant 2 : index
      %6 = upmem.pwram_alloc : memref<f32>
      %7 = arith.addi %5, %c64 : index
      %8 = arith.addi %7, %2 : index
      %9 = upmem.pwram_alloc : memref<64xf32>
      upmem.memcpy  mram_to_wram %4, %c64, %3 : memref<64xf32>, index, index
      upmem.memcpy  mram_to_wram %6, %c2, %5 : memref<f32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c64 step %c1 {
        %10 = memref.load %4[%arg0] : memref<64xf32>
        %11 = memref.load %6[] : memref<f32>
        %12 = arith.subf %10, %11 : f32
        memref.store %12, %9[%arg0] : memref<64xf32>
      }
      upmem.memcpy  wram_to_mram %9, %c64, %8 : memref<64xf32>, index, index
      upmem.return
    }
    upmem.func @softmax_262144_19() attributes {num_tasklets = 16 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c256 = arith.constant 256 : index
      %2 = arith.muli %1, %c256 : index
      %3 = arith.addi %0, %2 : index
      %c64 = arith.constant 64 : index
      %4 = upmem.pwram_alloc : memref<64xf32>
      %c4096 = arith.constant 4096 : index
      %5 = arith.addi %0, %c4096 : index
      %c2 = arith.constant 2 : index
      %6 = upmem.pwram_alloc : memref<f32>
      %7 = arith.addi %5, %c64 : index
      %8 = arith.addi %7, %2 : index
      %9 = upmem.pwram_alloc : memref<64xf32>
      upmem.memcpy  mram_to_wram %4, %c64, %3 : memref<64xf32>, index, index
      upmem.memcpy  mram_to_wram %6, %c2, %5 : memref<f32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c64 step %c1 {
        %10 = memref.load %4[%arg0] : memref<64xf32>
        %11 = memref.load %6[] : memref<f32>
        %12 = arith.divf %10, %11 : f32
        memref.store %12, %9[%arg0] : memref<64xf32>
      }
      upmem.memcpy  wram_to_mram %9, %c64, %8 : memref<64xf32>, index, index
      upmem.return
    }
    upmem.func @rmsnorm_262144_opt() attributes {num_tasklets = 16 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c256 = arith.constant 256 : index
      %2 = arith.muli %1, %c256 : index
      %3 = arith.addi %0, %2 : index
      %c64 = arith.constant 64 : index
      %4 = upmem.pwram_alloc : memref<64xf32>
      %c4096 = arith.constant 4096 : index
      %5 = arith.addi %0, %c4096 : index
      %6 = arith.addi %5, %2 : index
      %7 = upmem.pwram_alloc : memref<64xf32>
      %8 = arith.addi %5, %c4096 : index
      %c8 = arith.constant 8 : index
      %9 = arith.muli %1, %c8 : index
      %10 = arith.addi %8, %9 : index
      %c2 = arith.constant 2 : index
      %11 = upmem.pwram_alloc : memref<2xf32>
      upmem.memcpy  mram_to_wram %4, %c64, %3 : memref<64xf32>, index, index
      upmem.memcpy  mram_to_wram %7, %c64, %6 : memref<64xf32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cst = arith.constant 0.000000e+00 : f32
      memref.store %cst, %11[%c0] : memref<2xf32>
      memref.store %cst, %11[%c1] : memref<2xf32>
      scf.for %arg0 = %c0 to %c64 step %c2 {
        %12 = arith.addi %arg0, %c1 : index
        %13 = memref.load %4[%arg0] : memref<64xf32>
        %14 = memref.load %7[%arg0] : memref<64xf32>
        %15 = memref.load %11[%c0] : memref<2xf32>
        %16 = arith.mulf %13, %14 : f32
        %17 = arith.addf %16, %15 : f32
        memref.store %17, %11[%c0] : memref<2xf32>
        %18 = memref.load %4[%12] : memref<64xf32>
        %19 = memref.load %7[%12] : memref<64xf32>
        %20 = memref.load %11[%c1] : memref<2xf32>
        %21 = arith.mulf %18, %19 : f32
        %22 = arith.addf %21, %20 : f32
        memref.store %22, %11[%c1] : memref<2xf32>
      }
      upmem.memcpy  wram_to_mram %11, %c2, %10 : memref<2xf32>, index, index
      upmem.return
    }
    upmem.func @rmsnorm_262144_opt_20() attributes {num_tasklets = 16 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c256 = arith.constant 256 : index
      %2 = arith.muli %1, %c256 : index
      %3 = arith.addi %0, %2 : index
      %c64 = arith.constant 64 : index
      %4 = upmem.pwram_alloc : memref<64xf32>
      %c4096 = arith.constant 4096 : index
      %5 = arith.addi %0, %c4096 : index
      %6 = arith.addi %5, %2 : index
      %7 = upmem.pwram_alloc : memref<64xf32>
      %8 = arith.addi %5, %c4096 : index
      %c2 = arith.constant 2 : index
      %9 = upmem.pwram_alloc : memref<f32>
      %10 = arith.addi %8, %c64 : index
      %11 = arith.addi %10, %2 : index
      %12 = upmem.pwram_alloc : memref<64xf32>
      upmem.memcpy  mram_to_wram %4, %c64, %3 : memref<64xf32>, index, index
      upmem.memcpy  mram_to_wram %7, %c64, %6 : memref<64xf32>, index, index
      upmem.memcpy  mram_to_wram %9, %c2, %8 : memref<f32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %13 = memref.load %9[] : memref<f32>
      scf.for %arg0 = %c0 to %c64 step %c1 {
        %14 = memref.load %4[%arg0] : memref<64xf32>
        %15 = memref.load %7[%arg0] : memref<64xf32>
        %16 = arith.mulf %14, %15 : f32
        %17 = arith.mulf %16, %13 : f32
        memref.store %17, %12[%arg0] : memref<64xf32>
      }
      upmem.memcpy  wram_to_mram %12, %c64, %11 : memref<64xf32>, index, index
      upmem.return
    }
    upmem.func @softmax_262144_opt() attributes {num_tasklets = 16 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c256 = arith.constant 256 : index
      %2 = arith.muli %1, %c256 : index
      %3 = arith.addi %0, %2 : index
      %c64 = arith.constant 64 : index
      %4 = upmem.pwram_alloc : memref<64xf32>
      %c4096 = arith.constant 4096 : index
      %5 = arith.addi %0, %c4096 : index
      %c2 = arith.constant 2 : index
      %6 = upmem.pwram_alloc : memref<f32>
      %7 = arith.addi %5, %c64 : index
      %8 = arith.addi %7, %2 : index
      %9 = upmem.pwram_alloc : memref<64xf32>
      %10 = arith.addi %7, %c4096 : index
      %c8 = arith.constant 8 : index
      %11 = arith.muli %1, %c8 : index
      %12 = arith.addi %10, %11 : index
      %13 = upmem.pwram_alloc : memref<2xf32>
      upmem.memcpy  mram_to_wram %4, %c64, %3 : memref<64xf32>, index, index
      upmem.memcpy  mram_to_wram %6, %c2, %5 : memref<f32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %14 = memref.load %6[] : memref<f32>
      scf.for %arg0 = %c0 to %c64 step %c2 {
        %15 = arith.addi %arg0, %c1 : index
        %16 = memref.load %4[%arg0] : memref<64xf32>
        %17 = memref.load %13[%c0] : memref<2xf32>
        %18 = arith.subf %16, %14 : f32
        %19 = llvm.intr.exp(%18)  : (f32) -> f32
        %20 = arith.addf %17, %19 : f32
        memref.store %19, %9[%arg0] : memref<64xf32>
        memref.store %20, %13[%c0] : memref<2xf32>
        %21 = memref.load %4[%15] : memref<64xf32>
        %22 = memref.load %13[%c1] : memref<2xf32>
        %23 = arith.subf %21, %14 : f32
        %24 = llvm.intr.exp(%23)  : (f32) -> f32
        %25 = arith.addf %22, %24 : f32
        memref.store %19, %9[%15] : memref<64xf32>
        memref.store %25, %13[%c1] : memref<2xf32>
      }
      upmem.memcpy  wram_to_mram %9, %c64, %8 : memref<64xf32>, index, index
      upmem.memcpy  wram_to_mram %13, %c2, %12 : memref<2xf32>, index, index
      upmem.return
    }
    upmem.func @softmax_262144_opt_21() attributes {num_tasklets = 16 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c256 = arith.constant 256 : index
      %2 = arith.muli %1, %c256 : index
      %3 = arith.addi %0, %2 : index
      %c64 = arith.constant 64 : index
      %4 = upmem.pwram_alloc : memref<64xf32>
      %c4096 = arith.constant 4096 : index
      %5 = arith.addi %0, %c4096 : index
      %c2 = arith.constant 2 : index
      %6 = upmem.pwram_alloc : memref<f32>
      %7 = arith.addi %5, %c64 : index
      %8 = arith.addi %7, %2 : index
      %9 = upmem.pwram_alloc : memref<64xf32>
      upmem.memcpy  mram_to_wram %4, %c64, %3 : memref<64xf32>, index, index
      upmem.memcpy  mram_to_wram %6, %c2, %5 : memref<f32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %10 = memref.load %6[] : memref<f32>
      scf.for %arg0 = %c0 to %c64 step %c1 {
        %11 = memref.load %4[%arg0] : memref<64xf32>
        %12 = arith.divf %11, %10 : f32
        memref.store %12, %9[%arg0] : memref<64xf32>
      }
      upmem.memcpy  wram_to_mram %9, %c64, %8 : memref<64xf32>, index, index
      upmem.return
    }
    upmem.func @va_262144() attributes {num_tasklets = 16 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c256 = arith.constant 256 : index
      %2 = arith.muli %1, %c256 : index
      %3 = arith.addi %0, %2 : index
      %c64 = arith.constant 64 : index
      %4 = upmem.pwram_alloc : memref<64xf32>
      %c4096 = arith.constant 4096 : index
      %5 = arith.addi %0, %c4096 : index
      %6 = arith.addi %5, %2 : index
      %7 = upmem.pwram_alloc : memref<64xf32>
      %8 = arith.addi %5, %c4096 : index
      %9 = arith.addi %8, %2 : index
      %10 = upmem.pwram_alloc : memref<64xf32>
      upmem.memcpy  mram_to_wram %4, %c64, %3 : memref<64xf32>, index, index
      upmem.memcpy  mram_to_wram %7, %c64, %6 : memref<64xf32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c64 step %c1 {
        %11 = memref.load %4[%arg0] : memref<64xf32>
        %12 = memref.load %7[%arg0] : memref<64xf32>
        %13 = arith.addf %11, %12 : f32
        memref.store %13, %10[%arg0] : memref<64xf32>
      }
      upmem.memcpy  wram_to_mram %10, %c64, %9 : memref<64xf32>, index, index
      upmem.return
    }
    upmem.func @mha_big() attributes {num_tasklets = 16 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c64 = arith.constant 64 : index
      %2 = arith.muli %1, %c64 : index
      %3 = arith.addi %0, %2 : index
      %c16 = arith.constant 16 : index
      %4 = upmem.pwram_alloc : memref<16xf32>
      %c1024 = arith.constant 1024 : index
      %5 = arith.addi %0, %c1024 : index
      %6 = arith.addi %5, %2 : index
      %7 = upmem.pwram_alloc : memref<16xf32>
      %8 = arith.addi %5, %c1024 : index
      %9 = arith.addi %8, %2 : index
      %10 = upmem.pwram_alloc : memref<16xf32>
      upmem.memcpy  mram_to_wram %4, %c16, %3 : memref<16xf32>, index, index
      upmem.memcpy  mram_to_wram %7, %c16, %6 : memref<16xf32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c16 step %c1 {
        %11 = memref.load %4[%arg0] : memref<16xf32>
        %12 = memref.load %7[%arg0] : memref<16xf32>
        %13 = arith.mulf %11, %12 : f32
        memref.store %13, %10[%arg0] : memref<16xf32>
      }
      upmem.memcpy  wram_to_mram %10, %c16, %9 : memref<16xf32>, index, index
      upmem.return
    }
    upmem.func @mha_big_22() attributes {num_tasklets = 16 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c16 = arith.constant 16 : index
      %2 = arith.muli %1, %c16 : index
      %3 = arith.addi %0, %2 : index
      %c4 = arith.constant 4 : index
      %4 = upmem.pwram_alloc : memref<4xf32>
      %c256 = arith.constant 256 : index
      %5 = arith.addi %0, %c256 : index
      %c2 = arith.constant 2 : index
      %6 = upmem.pwram_alloc : memref<f32>
      %c64 = arith.constant 64 : index
      %7 = arith.addi %5, %c64 : index
      %8 = arith.addi %7, %2 : index
      %9 = upmem.pwram_alloc : memref<4xf32>
      upmem.memcpy  mram_to_wram %4, %c4, %3 : memref<4xf32>, index, index
      upmem.memcpy  mram_to_wram %6, %c2, %5 : memref<f32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4 step %c1 {
        %10 = memref.load %4[%arg0] : memref<4xf32>
        %11 = memref.load %6[] : memref<f32>
        %12 = arith.subf %10, %11 : f32
        memref.store %12, %9[%arg0] : memref<4xf32>
      }
      upmem.memcpy  wram_to_mram %9, %c4, %8 : memref<4xf32>, index, index
      upmem.return
    }
    upmem.func @mha_big_23() attributes {num_tasklets = 16 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c16 = arith.constant 16 : index
      %2 = arith.muli %1, %c16 : index
      %3 = arith.addi %0, %2 : index
      %c4 = arith.constant 4 : index
      %4 = upmem.pwram_alloc : memref<4xf32>
      %c256 = arith.constant 256 : index
      %5 = arith.addi %0, %c256 : index
      %c2 = arith.constant 2 : index
      %6 = upmem.pwram_alloc : memref<f32>
      %c64 = arith.constant 64 : index
      %7 = arith.addi %5, %c64 : index
      %8 = arith.addi %7, %2 : index
      %9 = upmem.pwram_alloc : memref<4xf32>
      upmem.memcpy  mram_to_wram %4, %c4, %3 : memref<4xf32>, index, index
      upmem.memcpy  mram_to_wram %6, %c2, %5 : memref<f32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4 step %c1 {
        %10 = memref.load %4[%arg0] : memref<4xf32>
        %11 = memref.load %6[] : memref<f32>
        %12 = arith.divf %10, %11 : f32
        memref.store %12, %9[%arg0] : memref<4xf32>
      }
      upmem.memcpy  wram_to_mram %9, %c4, %8 : memref<4xf32>, index, index
      upmem.return
    }
    upmem.func @mha_big_24() attributes {num_tasklets = 16 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c64 = arith.constant 64 : index
      %2 = arith.muli %1, %c64 : index
      %3 = arith.addi %0, %2 : index
      %c16 = arith.constant 16 : index
      %4 = upmem.pwram_alloc : memref<16xf32>
      %c1024 = arith.constant 1024 : index
      %5 = arith.addi %0, %c1024 : index
      %c2 = arith.constant 2 : index
      %6 = upmem.pwram_alloc : memref<f32>
      %7 = arith.addi %5, %c64 : index
      %8 = arith.addi %7, %2 : index
      %9 = upmem.pwram_alloc : memref<16xf32>
      upmem.memcpy  mram_to_wram %4, %c16, %3 : memref<16xf32>, index, index
      upmem.memcpy  mram_to_wram %6, %c2, %5 : memref<f32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c16 step %c1 {
        %10 = memref.load %4[%arg0] : memref<16xf32>
        %11 = memref.load %6[] : memref<f32>
        %12 = arith.mulf %10, %11 : f32
        memref.store %12, %9[%arg0] : memref<16xf32>
      }
      upmem.memcpy  wram_to_mram %9, %c16, %8 : memref<16xf32>, index, index
      upmem.return
    }
    upmem.func @mha_big_25() attributes {num_tasklets = 16 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c64 = arith.constant 64 : index
      %2 = arith.muli %1, %c64 : index
      %3 = arith.addi %0, %2 : index
      %c16 = arith.constant 16 : index
      %4 = upmem.pwram_alloc : memref<16xf32>
      %c1024 = arith.constant 1024 : index
      %5 = arith.addi %0, %c1024 : index
      %6 = arith.addi %5, %2 : index
      %7 = upmem.pwram_alloc : memref<16xf32>
      %8 = arith.addi %5, %c1024 : index
      %9 = arith.addi %8, %2 : index
      %10 = upmem.pwram_alloc : memref<16xf32>
      upmem.memcpy  mram_to_wram %4, %c16, %3 : memref<16xf32>, index, index
      upmem.memcpy  mram_to_wram %7, %c16, %6 : memref<16xf32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c16 step %c1 {
        %11 = memref.load %4[%arg0] : memref<16xf32>
        %12 = memref.load %7[%arg0] : memref<16xf32>
        %13 = arith.addf %11, %12 : f32
        memref.store %13, %10[%arg0] : memref<16xf32>
      }
      upmem.memcpy  wram_to_mram %10, %c16, %9 : memref<16xf32>, index, index
      upmem.return
    }
    upmem.func @test_0() attributes {num_tasklets = 16 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c16 = arith.constant 16 : index
      %2 = arith.muli %1, %c16 : index
      %3 = arith.addi %0, %2 : index
      %c4 = arith.constant 4 : index
      %4 = upmem.pwram_alloc : memref<4xi32>
      %c256 = arith.constant 256 : index
      %5 = arith.addi %0, %c256 : index
      %6 = arith.addi %5, %2 : index
      %7 = upmem.pwram_alloc : memref<4xi32>
      %8 = arith.addi %5, %c256 : index
      %9 = arith.addi %8, %2 : index
      %10 = upmem.pwram_alloc : memref<4xi32>
      upmem.memcpy  mram_to_wram %4, %c4, %3 : memref<4xi32>, index, index
      upmem.memcpy  mram_to_wram %7, %c4, %6 : memref<4xi32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4 step %c1 {
        %11 = memref.load %4[%arg0] : memref<4xi32>
        %12 = memref.load %7[%arg0] : memref<4xi32>
        %13 = arith.addi %11, %12 : i32
        memref.store %13, %10[%arg0] : memref<4xi32>
      }
      upmem.memcpy  wram_to_mram %10, %c4, %9 : memref<4xi32>, index, index
      upmem.return
    }
    upmem.func @test_2() attributes {num_tasklets = 16 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c16 = arith.constant 16 : index
      %2 = arith.muli %1, %c16 : index
      %3 = arith.addi %0, %2 : index
      %c4 = arith.constant 4 : index
      %4 = upmem.pwram_alloc : memref<4xi32>
      %c256 = arith.constant 256 : index
      %5 = arith.addi %0, %c256 : index
      %6 = arith.addi %5, %2 : index
      %7 = upmem.pwram_alloc : memref<4xi32>
      %8 = arith.addi %5, %c256 : index
      %9 = arith.addi %8, %2 : index
      %10 = upmem.pwram_alloc : memref<4xi32>
      %11 = arith.addi %8, %c256 : index
      %12 = arith.addi %11, %2 : index
      %13 = upmem.pwram_alloc : memref<4xi32>
      %14 = arith.addi %11, %c256 : index
      %15 = arith.addi %14, %2 : index
      %16 = upmem.pwram_alloc : memref<4xi32>
      %17 = arith.addi %14, %c256 : index
      %18 = arith.addi %17, %2 : index
      %19 = upmem.pwram_alloc : memref<4xi32>
      %20 = arith.addi %17, %c256 : index
      %21 = arith.addi %20, %2 : index
      %22 = upmem.pwram_alloc : memref<4xi32>
      %23 = arith.addi %20, %c256 : index
      %24 = arith.addi %23, %2 : index
      %25 = upmem.pwram_alloc : memref<4xi32>
      %26 = arith.addi %23, %c256 : index
      %27 = arith.addi %26, %2 : index
      %28 = upmem.pwram_alloc : memref<4xi32>
      upmem.memcpy  mram_to_wram %4, %c4, %3 : memref<4xi32>, index, index
      upmem.memcpy  mram_to_wram %7, %c4, %6 : memref<4xi32>, index, index
      upmem.memcpy  mram_to_wram %10, %c4, %9 : memref<4xi32>, index, index
      upmem.memcpy  mram_to_wram %13, %c4, %12 : memref<4xi32>, index, index
      upmem.memcpy  mram_to_wram %16, %c4, %15 : memref<4xi32>, index, index
      upmem.memcpy  mram_to_wram %19, %c4, %18 : memref<4xi32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4 step %c1 {
        %29 = memref.load %4[%arg0] : memref<4xi32>
        %30 = memref.load %7[%arg0] : memref<4xi32>
        %31 = arith.addi %29, %30 : i32
        memref.store %31, %22[%arg0] : memref<4xi32>
      }
      scf.for %arg0 = %c0 to %c4 step %c1 {
        %29 = memref.load %10[%arg0] : memref<4xi32>
        %30 = memref.load %13[%arg0] : memref<4xi32>
        %31 = arith.addi %29, %30 : i32
        memref.store %31, %25[%arg0] : memref<4xi32>
      }
      scf.for %arg0 = %c0 to %c4 step %c1 {
        %29 = memref.load %16[%arg0] : memref<4xi32>
        %30 = memref.load %19[%arg0] : memref<4xi32>
        %31 = arith.addi %29, %30 : i32
        memref.store %31, %28[%arg0] : memref<4xi32>
      }
      upmem.memcpy  wram_to_mram %22, %c4, %21 : memref<4xi32>, index, index
      upmem.memcpy  wram_to_mram %25, %c4, %24 : memref<4xi32>, index, index
      upmem.memcpy  wram_to_mram %28, %c4, %27 : memref<4xi32>, index, index
      upmem.return
    }
  }
}

