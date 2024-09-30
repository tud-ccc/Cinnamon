#map = affine_map<(d0, d1, d2) -> (d1 * 8 + d2)>
#map1 = affine_map<(d0, d1, d2) -> (0)>
#map2 = affine_map<(d0, d1, d2) -> (d1 * 8 + d2, 0)>
#map3 = affine_map<(d0, d1, d2) -> (d1 * 16 + d2)>
#map4 = affine_map<(d0, d1, d2) -> (d1 * 16 + d2, 0)>
#map5 = affine_map<(d0, d1, d2) -> (d2)>
#map6 = affine_map<(d0, d1, d2) -> (d0 * 1024 + d1 * 16 + d2)>
module {
  memref.global "private" constant @__constant_1xi64_7 : memref<1xi64> = dense<32768> {alignment = 64 : i64}
  memref.global "private" constant @__constant_256x1xf32 : memref<256x1xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_2xi64_8 : memref<2xi64> = dense<[768, 1]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1xi64_6 : memref<1xi64> = dense<768> {alignment = 64 : i64}
  memref.global "private" constant @__constant_48x6xf32 : memref<48x6xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_2xi64_7 : memref<2xi64> = dense<[48, 6]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_48x1xf32 : memref<48x1xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_2xi64_6 : memref<2xi64> = dense<[288, 1]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1xi64_5 : memref<1xi64> = dense<1024> {alignment = 64 : i64}
  memref.global "private" constant @__constant_256x4xf32 : memref<256x4xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_2xi64_5 : memref<2xi64> = dense<[256, 4]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1024xf32 : memref<1024xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1xi64_4 : memref<1xi64> = dense<4096> {alignment = 64 : i64}
  memref.global "private" constant @__constant_256x16xf32 : memref<256x16xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_2xi64_4 : memref<2xi64> = dense<[256, 16]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_4096xf32 : memref<4096xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1xi64_3 : memref<1xi64> = dense<48> {alignment = 64 : i64}
  memref.global "private" constant @__constant_8x6xf32 : memref<8x6xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_2xi64_3 : memref<2xi64> = dense<[8, 6]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_48xf32 : memref<48xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1xi64_2 : memref<1xi64> = dense<256> {alignment = 64 : i64}
  memref.global "private" constant @__constant_128x2xf32 : memref<128x2xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_2xi64_2 : memref<2xi64> = dense<[128, 2]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_256xf32 : memref<256xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1xi64_1 : memref<1xi64> = dense<288> {alignment = 64 : i64}
  memref.global "private" constant @__constant_16x18xf32 : memref<16x18xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_2xi64_1 : memref<2xi64> = dense<[16, 18]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_288xf32 : memref<288xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_xf32_0 : memref<f32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1xi64_0 : memref<1xi64> = dense<262144> {alignment = 64 : i64}
  memref.global "private" constant @__constant_4096x64xf32 : memref<4096x64xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_4096x2xf32 : memref<4096x2xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_2xi64_0 : memref<2xi64> = dense<[4096, 64]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_262144xf32 : memref<262144xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_xf32 : memref<f32> = dense<0xFF800000> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1xi64 : memref<1xi64> = dense<1048576> {alignment = 64 : i64}
  memref.global "private" constant @__constant_4096x256xf32 : memref<4096x256xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_2xi64 : memref<2xi64> = dense<[4096, 256]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1048576xf32 : memref<1048576xf32> = dense<0.000000e+00> {alignment = 64 : i64}
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
        %33 = memref.get_global @__constant_48x1xf32 : memref<48x1xf32>
        %subview_82 = memref.subview %subview_53[%arg18, 0] [48, 288] [1, 1] : memref<288x288xf32, strided<[288, 1], offset: ?>> to memref<48x288xf32, strided<[288, 1], offset: ?>>
        %34 = cnm.workgroup : !cnm.workgroup<1x6x8>
        scf.for %arg20 = %c0 to %c288 step %c1 {
          %38 = memref.load %reshape_56[%arg20, %c0] : memref<288x1xf32>
          memref.store %38, %alloc_7[%c0, %arg20] : memref<1x288xf32>
        }
        %35 = cnm.alloc() for %34 : !cnm.buffer<288xf32 on 1x6x8, level 0>
        %36 = cnm.alloc() for %34 : !cnm.buffer<288xf32 on 1x6x8, level 0>
        %37 = cnm.alloc() for %34 : !cnm.buffer<f32 on 1x6x8, level 0>
        cnm.scatter %subview_82 into %35[#map] of %34 : memref<48x288xf32, strided<[288, 1], offset: ?>> into !cnm.buffer<288xf32 on 1x6x8, level 0>
        cnm.scatter %alloc_7 into %36[#map1] of %34 : memref<1x288xf32> into !cnm.buffer<288xf32 on 1x6x8, level 0>
        cnm.scatter %33 into %37[#map2] of %34 : memref<48x1xf32> into !cnm.buffer<f32 on 1x6x8, level 0>
        cnm.launch %34 in(%35, %36 : !cnm.buffer<288xf32 on 1x6x8, level 0>, !cnm.buffer<288xf32 on 1x6x8, level 0>) out(%37 : !cnm.buffer<f32 on 1x6x8, level 0>) on !cnm.workgroup<1x6x8> {
        ^bb0(%arg20: memref<288xf32>, %arg21: memref<288xf32>, %arg22: memref<f32>):
          %c0_85 = arith.constant 0 : index
          %c288_86 = arith.constant 288 : index
          %c1_87 = arith.constant 1 : index
          scf.for %arg23 = %c0_85 to %c288_86 step %c1_87 {
            %38 = memref.load %arg20[%arg23] : memref<288xf32>
            %39 = memref.load %arg21[%arg23] : memref<288xf32>
            %40 = memref.load %arg22[] : memref<f32>
            %41 = arith.mulf %38, %39 : f32
            %42 = arith.addf %41, %40 : f32
            memref.store %42, %arg22[] : memref<f32>
          }
        }
        cnm.gather %37[#map2] of %34 into %alloc_8 : !cnm.buffer<f32 on 1x6x8, level 0> into memref<48x1xf32>
        cnm.free_workgroup %34 : !cnm.workgroup<1x6x8>
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
        %33 = memref.get_global @__constant_48x1xf32 : memref<48x1xf32>
        %subview_82 = memref.subview %subview_54[%arg18, 0] [48, 288] [1, 1] : memref<288x288xf32, strided<[288, 1], offset: ?>> to memref<48x288xf32, strided<[288, 1], offset: ?>>
        %34 = cnm.workgroup : !cnm.workgroup<1x6x8>
        scf.for %arg20 = %c0 to %c288 step %c1 {
          %38 = memref.load %reshape_56[%arg20, %c0] : memref<288x1xf32>
          memref.store %38, %alloc_11[%c0, %arg20] : memref<1x288xf32>
        }
        %35 = cnm.alloc() for %34 : !cnm.buffer<288xf32 on 1x6x8, level 0>
        %36 = cnm.alloc() for %34 : !cnm.buffer<288xf32 on 1x6x8, level 0>
        %37 = cnm.alloc() for %34 : !cnm.buffer<f32 on 1x6x8, level 0>
        cnm.scatter %subview_82 into %35[#map] of %34 : memref<48x288xf32, strided<[288, 1], offset: ?>> into !cnm.buffer<288xf32 on 1x6x8, level 0>
        cnm.scatter %alloc_11 into %36[#map1] of %34 : memref<1x288xf32> into !cnm.buffer<288xf32 on 1x6x8, level 0>
        cnm.scatter %33 into %37[#map2] of %34 : memref<48x1xf32> into !cnm.buffer<f32 on 1x6x8, level 0>
        cnm.launch %34 in(%35, %36 : !cnm.buffer<288xf32 on 1x6x8, level 0>, !cnm.buffer<288xf32 on 1x6x8, level 0>) out(%37 : !cnm.buffer<f32 on 1x6x8, level 0>) on !cnm.workgroup<1x6x8> {
        ^bb0(%arg20: memref<288xf32>, %arg21: memref<288xf32>, %arg22: memref<f32>):
          %c0_85 = arith.constant 0 : index
          %c288_86 = arith.constant 288 : index
          %c1_87 = arith.constant 1 : index
          scf.for %arg23 = %c0_85 to %c288_86 step %c1_87 {
            %38 = memref.load %arg20[%arg23] : memref<288xf32>
            %39 = memref.load %arg21[%arg23] : memref<288xf32>
            %40 = memref.load %arg22[] : memref<f32>
            %41 = arith.mulf %38, %39 : f32
            %42 = arith.addf %41, %40 : f32
            memref.store %42, %arg22[] : memref<f32>
          }
        }
        cnm.gather %37[#map2] of %34 into %alloc_12 : !cnm.buffer<f32 on 1x6x8, level 0> into memref<48x1xf32>
        cnm.free_workgroup %34 : !cnm.workgroup<1x6x8>
        %alloc_83 = memref.alloc() : memref<288x1xf32>
        memref.copy %arg19, %alloc_83 : memref<288x1xf32> to memref<288x1xf32>
        %subview_84 = memref.subview %alloc_83[%arg18, 0] [48, 1] [1, 1] : memref<288x1xf32> to memref<48x1xf32, strided<[1, 1], offset: ?>>
        memref.copy %alloc_12, %subview_84 : memref<48x1xf32> to memref<48x1xf32, strided<[1, 1], offset: ?>>
        scf.yield %alloc_83 : memref<288x1xf32>
      }
      %reshape_58 = memref.reshape %9(%8) : (memref<288x1xf32>, memref<1xi64>) -> memref<288xf32>
      memref.copy %alloc_13, %alloc_14 : memref<288x1xf32> to memref<288x1xf32>
      %10 = scf.for %arg18 = %c0 to %c288 step %c48 iter_args(%arg19 = %alloc_14) -> (memref<288x1xf32>) {
        %33 = memref.get_global @__constant_48x1xf32 : memref<48x1xf32>
        %subview_82 = memref.subview %subview_55[%arg18, 0] [48, 288] [1, 1] : memref<288x288xf32, strided<[288, 1], offset: ?>> to memref<48x288xf32, strided<[288, 1], offset: ?>>
        %34 = cnm.workgroup : !cnm.workgroup<1x6x8>
        scf.for %arg20 = %c0 to %c288 step %c1 {
          %38 = memref.load %reshape_56[%arg20, %c0] : memref<288x1xf32>
          memref.store %38, %alloc_15[%c0, %arg20] : memref<1x288xf32>
        }
        %35 = cnm.alloc() for %34 : !cnm.buffer<288xf32 on 1x6x8, level 0>
        %36 = cnm.alloc() for %34 : !cnm.buffer<288xf32 on 1x6x8, level 0>
        %37 = cnm.alloc() for %34 : !cnm.buffer<f32 on 1x6x8, level 0>
        cnm.scatter %subview_82 into %35[#map] of %34 : memref<48x288xf32, strided<[288, 1], offset: ?>> into !cnm.buffer<288xf32 on 1x6x8, level 0>
        cnm.scatter %alloc_15 into %36[#map1] of %34 : memref<1x288xf32> into !cnm.buffer<288xf32 on 1x6x8, level 0>
        cnm.scatter %33 into %37[#map2] of %34 : memref<48x1xf32> into !cnm.buffer<f32 on 1x6x8, level 0>
        cnm.launch %34 in(%35, %36 : !cnm.buffer<288xf32 on 1x6x8, level 0>, !cnm.buffer<288xf32 on 1x6x8, level 0>) out(%37 : !cnm.buffer<f32 on 1x6x8, level 0>) on !cnm.workgroup<1x6x8> {
        ^bb0(%arg20: memref<288xf32>, %arg21: memref<288xf32>, %arg22: memref<f32>):
          %c0_85 = arith.constant 0 : index
          %c288_86 = arith.constant 288 : index
          %c1_87 = arith.constant 1 : index
          scf.for %arg23 = %c0_85 to %c288_86 step %c1_87 {
            %38 = memref.load %arg20[%arg23] : memref<288xf32>
            %39 = memref.load %arg21[%arg23] : memref<288xf32>
            %40 = memref.load %arg22[] : memref<f32>
            %41 = arith.mulf %38, %39 : f32
            %42 = arith.addf %41, %40 : f32
            memref.store %42, %arg22[] : memref<f32>
          }
        }
        cnm.gather %37[#map2] of %34 into %alloc_16 : !cnm.buffer<f32 on 1x6x8, level 0> into memref<48x1xf32>
        cnm.free_workgroup %34 : !cnm.workgroup<1x6x8>
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
        %33 = arith.remui %arg18, %c48 : index
        %34 = arith.index_cast %33 : index to i64
        %35 = arith.uitofp %34 : i64 to f32
        %36 = arith.divf %35, %cst_1 : f32
        %37 = math.powf %cst_2, %36 : f32
        %38 = arith.divf %cst_0, %37 : f32
        %39 = arith.mulf %12, %38 : f32
        %40 = math.cos %39 : f32
        %41 = math.sin %39 : f32
        %alloc_82 = memref.alloc() : memref<288xf32>
        memref.copy %arg19, %alloc_82 : memref<288xf32> to memref<288xf32>
        %42 = func.call @rot(%alloc_82, %arg18, %40, %41) : (memref<288xf32>, index, f32, f32) -> memref<288xf32>
        %43 = arith.cmpi ult, %arg18, %c288 : index
        %alloc_83 = memref.alloc() : memref<288xf32>
        %44 = scf.if %43 -> (memref<288xf32>) {
          memref.copy %arg20, %alloc_83 : memref<288xf32> to memref<288xf32>
          %45 = func.call @rot(%alloc_83, %arg18, %40, %41) : (memref<288xf32>, index, f32, f32) -> memref<288xf32>
          scf.yield %45 : memref<288xf32>
        } else {
          scf.yield %arg20 : memref<288xf32>
        }
        scf.yield %42, %44 : memref<288xf32>, memref<288xf32>
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
        %33 = memref.get_global @__constant_48x1xf32 : memref<48x1xf32>
        %subview_82 = memref.subview %subview_64[%arg18, 0] [48, 288] [1, 1] : memref<288x288xf32, strided<[288, 1], offset: ?>> to memref<48x288xf32, strided<[288, 1], offset: ?>>
        %34 = cnm.workgroup : !cnm.workgroup<1x6x8>
        scf.for %arg20 = %c0 to %c288 step %c1 {
          %38 = memref.load %reshape_65[%arg20, %c0] : memref<288x1xf32>
          memref.store %38, %alloc_24[%c0, %arg20] : memref<1x288xf32>
        }
        %35 = cnm.alloc() for %34 : !cnm.buffer<288xf32 on 1x6x8, level 0>
        %36 = cnm.alloc() for %34 : !cnm.buffer<288xf32 on 1x6x8, level 0>
        %37 = cnm.alloc() for %34 : !cnm.buffer<f32 on 1x6x8, level 0>
        cnm.scatter %subview_82 into %35[#map] of %34 : memref<48x288xf32, strided<[288, 1], offset: ?>> into !cnm.buffer<288xf32 on 1x6x8, level 0>
        cnm.scatter %alloc_24 into %36[#map1] of %34 : memref<1x288xf32> into !cnm.buffer<288xf32 on 1x6x8, level 0>
        cnm.scatter %33 into %37[#map2] of %34 : memref<48x1xf32> into !cnm.buffer<f32 on 1x6x8, level 0>
        cnm.launch %34 in(%35, %36 : !cnm.buffer<288xf32 on 1x6x8, level 0>, !cnm.buffer<288xf32 on 1x6x8, level 0>) out(%37 : !cnm.buffer<f32 on 1x6x8, level 0>) on !cnm.workgroup<1x6x8> {
        ^bb0(%arg20: memref<288xf32>, %arg21: memref<288xf32>, %arg22: memref<f32>):
          %c0_85 = arith.constant 0 : index
          %c288_86 = arith.constant 288 : index
          %c1_87 = arith.constant 1 : index
          scf.for %arg23 = %c0_85 to %c288_86 step %c1_87 {
            %38 = memref.load %arg20[%arg23] : memref<288xf32>
            %39 = memref.load %arg21[%arg23] : memref<288xf32>
            %40 = memref.load %arg22[] : memref<f32>
            %41 = arith.mulf %38, %39 : f32
            %42 = arith.addf %41, %40 : f32
            memref.store %42, %arg22[] : memref<f32>
          }
        }
        cnm.gather %37[#map2] of %34 into %alloc_25 : !cnm.buffer<f32 on 1x6x8, level 0> into memref<48x1xf32>
        cnm.free_workgroup %34 : !cnm.workgroup<1x6x8>
        %alloc_83 = memref.alloc() : memref<288x1xf32>
        memref.copy %arg19, %alloc_83 : memref<288x1xf32> to memref<288x1xf32>
        %subview_84 = memref.subview %alloc_83[%arg18, 0] [48, 1] [1, 1] : memref<288x1xf32> to memref<48x1xf32, strided<[1, 1], offset: ?>>
        memref.copy %alloc_25, %subview_84 : memref<48x1xf32> to memref<48x1xf32, strided<[1, 1], offset: ?>>
        scf.yield %alloc_83 : memref<288x1xf32>
      }
      %reshape_66 = memref.reshape %15(%8) : (memref<288x1xf32>, memref<1xi64>) -> memref<288xf32>
      %16 = cnm.workgroup : !cnm.workgroup<1x6x8>
      %17 = memref.get_global @__constant_2xi64_7 : memref<2xi64>
      %reshape_67 = memref.reshape %arg17(%17) : (memref<288xf32>, memref<2xi64>) -> memref<48x6xf32>
      %18 = cnm.alloc() for %16 : !cnm.buffer<6xf32 on 1x6x8, level 0>
      cnm.scatter %reshape_67 into %18[#map] of %16 : memref<48x6xf32> into !cnm.buffer<6xf32 on 1x6x8, level 0>
      %reshape_68 = memref.reshape %reshape_66(%17) : (memref<288xf32>, memref<2xi64>) -> memref<48x6xf32>
      %19 = cnm.alloc() for %16 : !cnm.buffer<6xf32 on 1x6x8, level 0>
      cnm.scatter %reshape_68 into %19[#map] of %16 : memref<48x6xf32> into !cnm.buffer<6xf32 on 1x6x8, level 0>
      %20 = memref.get_global @__constant_48x6xf32 : memref<48x6xf32>
      %21 = cnm.alloc() for %16 : !cnm.buffer<6xf32 on 1x6x8, level 0>
      cnm.scatter %20 into %21[#map] of %16 : memref<48x6xf32> into !cnm.buffer<6xf32 on 1x6x8, level 0>
      cnm.launch %16 in(%18, %19 : !cnm.buffer<6xf32 on 1x6x8, level 0>, !cnm.buffer<6xf32 on 1x6x8, level 0>) out(%21 : !cnm.buffer<6xf32 on 1x6x8, level 0>) on !cnm.workgroup<1x6x8> {
      ^bb0(%arg18: memref<6xf32>, %arg19: memref<6xf32>, %arg20: memref<6xf32>):
        %c0_82 = arith.constant 0 : index
        %c6_83 = arith.constant 6 : index
        %c1_84 = arith.constant 1 : index
        scf.for %arg21 = %c0_82 to %c6_83 step %c1_84 {
          %33 = memref.load %arg18[%arg21] : memref<6xf32>
          %34 = memref.load %arg19[%arg21] : memref<6xf32>
          %35 = arith.addf %33, %34 : f32
          memref.store %35, %arg20[%arg21] : memref<6xf32>
        }
      }
      cnm.gather %21[#map] of %16 into %alloc_26 : !cnm.buffer<6xf32 on 1x6x8, level 0> into memref<48x6xf32>
      %reshape_69 = memref.reshape %alloc_26(%8) : (memref<48x6xf32>, memref<1xi64>) -> memref<288xf32>
      cnm.free_workgroup %16 : !cnm.workgroup<1x6x8>
      %subview_70 = memref.subview %arg13[%arg16, 0] [1, 288] [1, 1] : memref<6x288xf32> to memref<288xf32, strided<[1], offset: ?>>
      memref.copy %reshape_69, %alloc_27 : memref<288xf32> to memref<288xf32>
      memref.copy %subview_70, %alloc_28 : memref<288xf32, strided<[1], offset: ?>> to memref<288xf32>
      %22 = func.call @rmsnorm(%alloc_27, %alloc_28) : (memref<288xf32>, memref<288xf32>) -> memref<288xf32>
      %subview_71 = memref.subview %arg10[%arg16, 0, 0] [1, 768, 288] [1, 1, 1] : memref<6x768x288xf32> to memref<768x288xf32, strided<[288, 1], offset: ?>>
      %subview_72 = memref.subview %arg12[%arg16, 0, 0] [1, 768, 288] [1, 1, 1] : memref<6x768x288xf32> to memref<768x288xf32, strided<[288, 1], offset: ?>>
      %reshape_73 = memref.reshape %22(%6) : (memref<288xf32>, memref<2xi64>) -> memref<288x1xf32>
      memref.copy %alloc_29, %alloc_30 : memref<768x1xf32> to memref<768x1xf32>
      %23 = scf.for %arg18 = %c0 to %c768 step %c48 iter_args(%arg19 = %alloc_30) -> (memref<768x1xf32>) {
        %33 = memref.get_global @__constant_48x1xf32 : memref<48x1xf32>
        %subview_82 = memref.subview %subview_71[%arg18, 0] [48, 288] [1, 1] : memref<768x288xf32, strided<[288, 1], offset: ?>> to memref<48x288xf32, strided<[288, 1], offset: ?>>
        %34 = cnm.workgroup : !cnm.workgroup<1x6x8>
        scf.for %arg20 = %c0 to %c288 step %c1 {
          %38 = memref.load %reshape_73[%arg20, %c0] : memref<288x1xf32>
          memref.store %38, %alloc_31[%c0, %arg20] : memref<1x288xf32>
        }
        %35 = cnm.alloc() for %34 : !cnm.buffer<288xf32 on 1x6x8, level 0>
        %36 = cnm.alloc() for %34 : !cnm.buffer<288xf32 on 1x6x8, level 0>
        %37 = cnm.alloc() for %34 : !cnm.buffer<f32 on 1x6x8, level 0>
        cnm.scatter %subview_82 into %35[#map] of %34 : memref<48x288xf32, strided<[288, 1], offset: ?>> into !cnm.buffer<288xf32 on 1x6x8, level 0>
        cnm.scatter %alloc_31 into %36[#map1] of %34 : memref<1x288xf32> into !cnm.buffer<288xf32 on 1x6x8, level 0>
        cnm.scatter %33 into %37[#map2] of %34 : memref<48x1xf32> into !cnm.buffer<f32 on 1x6x8, level 0>
        cnm.launch %34 in(%35, %36 : !cnm.buffer<288xf32 on 1x6x8, level 0>, !cnm.buffer<288xf32 on 1x6x8, level 0>) out(%37 : !cnm.buffer<f32 on 1x6x8, level 0>) on !cnm.workgroup<1x6x8> {
        ^bb0(%arg20: memref<288xf32>, %arg21: memref<288xf32>, %arg22: memref<f32>):
          %c0_85 = arith.constant 0 : index
          %c288_86 = arith.constant 288 : index
          %c1_87 = arith.constant 1 : index
          scf.for %arg23 = %c0_85 to %c288_86 step %c1_87 {
            %38 = memref.load %arg20[%arg23] : memref<288xf32>
            %39 = memref.load %arg21[%arg23] : memref<288xf32>
            %40 = memref.load %arg22[] : memref<f32>
            %41 = arith.mulf %38, %39 : f32
            %42 = arith.addf %41, %40 : f32
            memref.store %42, %arg22[] : memref<f32>
          }
        }
        cnm.gather %37[#map2] of %34 into %alloc_32 : !cnm.buffer<f32 on 1x6x8, level 0> into memref<48x1xf32>
        cnm.free_workgroup %34 : !cnm.workgroup<1x6x8>
        %alloc_83 = memref.alloc() : memref<768x1xf32>
        memref.copy %arg19, %alloc_83 : memref<768x1xf32> to memref<768x1xf32>
        %subview_84 = memref.subview %alloc_83[%arg18, 0] [48, 1] [1, 1] : memref<768x1xf32> to memref<48x1xf32, strided<[1, 1], offset: ?>>
        memref.copy %alloc_32, %subview_84 : memref<48x1xf32> to memref<48x1xf32, strided<[1, 1], offset: ?>>
        scf.yield %alloc_83 : memref<768x1xf32>
      }
      %24 = memref.get_global @__constant_1xi64_6 : memref<1xi64>
      %reshape_74 = memref.reshape %23(%24) : (memref<768x1xf32>, memref<1xi64>) -> memref<768xf32>
      memref.copy %alloc_33, %alloc_34 : memref<768x1xf32> to memref<768x1xf32>
      %25 = scf.for %arg18 = %c0 to %c768 step %c48 iter_args(%arg19 = %alloc_34) -> (memref<768x1xf32>) {
        %33 = memref.get_global @__constant_48x1xf32 : memref<48x1xf32>
        %subview_82 = memref.subview %subview_72[%arg18, 0] [48, 288] [1, 1] : memref<768x288xf32, strided<[288, 1], offset: ?>> to memref<48x288xf32, strided<[288, 1], offset: ?>>
        %34 = cnm.workgroup : !cnm.workgroup<1x6x8>
        scf.for %arg20 = %c0 to %c288 step %c1 {
          %38 = memref.load %reshape_73[%arg20, %c0] : memref<288x1xf32>
          memref.store %38, %alloc_35[%c0, %arg20] : memref<1x288xf32>
        }
        %35 = cnm.alloc() for %34 : !cnm.buffer<288xf32 on 1x6x8, level 0>
        %36 = cnm.alloc() for %34 : !cnm.buffer<288xf32 on 1x6x8, level 0>
        %37 = cnm.alloc() for %34 : !cnm.buffer<f32 on 1x6x8, level 0>
        cnm.scatter %subview_82 into %35[#map] of %34 : memref<48x288xf32, strided<[288, 1], offset: ?>> into !cnm.buffer<288xf32 on 1x6x8, level 0>
        cnm.scatter %alloc_35 into %36[#map1] of %34 : memref<1x288xf32> into !cnm.buffer<288xf32 on 1x6x8, level 0>
        cnm.scatter %33 into %37[#map2] of %34 : memref<48x1xf32> into !cnm.buffer<f32 on 1x6x8, level 0>
        cnm.launch %34 in(%35, %36 : !cnm.buffer<288xf32 on 1x6x8, level 0>, !cnm.buffer<288xf32 on 1x6x8, level 0>) out(%37 : !cnm.buffer<f32 on 1x6x8, level 0>) on !cnm.workgroup<1x6x8> {
        ^bb0(%arg20: memref<288xf32>, %arg21: memref<288xf32>, %arg22: memref<f32>):
          %c0_85 = arith.constant 0 : index
          %c288_86 = arith.constant 288 : index
          %c1_87 = arith.constant 1 : index
          scf.for %arg23 = %c0_85 to %c288_86 step %c1_87 {
            %38 = memref.load %arg20[%arg23] : memref<288xf32>
            %39 = memref.load %arg21[%arg23] : memref<288xf32>
            %40 = memref.load %arg22[] : memref<f32>
            %41 = arith.mulf %38, %39 : f32
            %42 = arith.addf %41, %40 : f32
            memref.store %42, %arg22[] : memref<f32>
          }
        }
        cnm.gather %37[#map2] of %34 into %alloc_36 : !cnm.buffer<f32 on 1x6x8, level 0> into memref<48x1xf32>
        cnm.free_workgroup %34 : !cnm.workgroup<1x6x8>
        %alloc_83 = memref.alloc() : memref<768x1xf32>
        memref.copy %arg19, %alloc_83 : memref<768x1xf32> to memref<768x1xf32>
        %subview_84 = memref.subview %alloc_83[%arg18, 0] [48, 1] [1, 1] : memref<768x1xf32> to memref<48x1xf32, strided<[1, 1], offset: ?>>
        memref.copy %alloc_36, %subview_84 : memref<48x1xf32> to memref<48x1xf32, strided<[1, 1], offset: ?>>
        scf.yield %alloc_83 : memref<768x1xf32>
      }
      %reshape_75 = memref.reshape %25(%24) : (memref<768x1xf32>, memref<1xi64>) -> memref<768xf32>
      memref.copy %reshape_74, %alloc_37 : memref<768xf32> to memref<768xf32>
      %26 = scf.for %arg18 = %c0 to %c768 step %c1 iter_args(%arg19 = %alloc_37) -> (memref<768xf32>) {
        %33 = memref.load %arg19[%arg18] : memref<768xf32>
        %34 = memref.load %reshape_75[%arg18] : memref<768xf32>
        %35 = math.exp %33 : f32
        %36 = arith.addf %35, %cst_0 : f32
        %37 = arith.divf %cst_0, %36 : f32
        %38 = arith.mulf %34, %37 : f32
        %alloc_82 = memref.alloc() : memref<768xf32>
        memref.copy %arg19, %alloc_82 : memref<768xf32> to memref<768xf32>
        memref.store %38, %alloc_82[%arg18] : memref<768xf32>
        scf.yield %alloc_82 : memref<768xf32>
      }
      %subview_76 = memref.subview %arg11[%arg16, 0, 0] [1, 288, 768] [1, 1, 1] : memref<6x288x768xf32> to memref<288x768xf32, strided<[768, 1], offset: ?>>
      %27 = memref.get_global @__constant_2xi64_8 : memref<2xi64>
      %reshape_77 = memref.reshape %26(%27) : (memref<768xf32>, memref<2xi64>) -> memref<768x1xf32>
      memref.copy %alloc_38, %alloc_39 : memref<288x1xf32> to memref<288x1xf32>
      %28 = scf.for %arg18 = %c0 to %c288 step %c48 iter_args(%arg19 = %alloc_39) -> (memref<288x1xf32>) {
        %33 = memref.get_global @__constant_48x1xf32 : memref<48x1xf32>
        %subview_82 = memref.subview %subview_76[%arg18, 0] [48, 768] [1, 1] : memref<288x768xf32, strided<[768, 1], offset: ?>> to memref<48x768xf32, strided<[768, 1], offset: ?>>
        %34 = cnm.workgroup : !cnm.workgroup<1x6x8>
        scf.for %arg20 = %c0 to %c768 step %c1 {
          %38 = memref.load %reshape_77[%arg20, %c0] : memref<768x1xf32>
          memref.store %38, %alloc_40[%c0, %arg20] : memref<1x768xf32>
        }
        %35 = cnm.alloc() for %34 : !cnm.buffer<768xf32 on 1x6x8, level 0>
        %36 = cnm.alloc() for %34 : !cnm.buffer<768xf32 on 1x6x8, level 0>
        %37 = cnm.alloc() for %34 : !cnm.buffer<f32 on 1x6x8, level 0>
        cnm.scatter %subview_82 into %35[#map] of %34 : memref<48x768xf32, strided<[768, 1], offset: ?>> into !cnm.buffer<768xf32 on 1x6x8, level 0>
        cnm.scatter %alloc_40 into %36[#map1] of %34 : memref<1x768xf32> into !cnm.buffer<768xf32 on 1x6x8, level 0>
        cnm.scatter %33 into %37[#map2] of %34 : memref<48x1xf32> into !cnm.buffer<f32 on 1x6x8, level 0>
        cnm.launch %34 in(%35, %36 : !cnm.buffer<768xf32 on 1x6x8, level 0>, !cnm.buffer<768xf32 on 1x6x8, level 0>) out(%37 : !cnm.buffer<f32 on 1x6x8, level 0>) on !cnm.workgroup<1x6x8> {
        ^bb0(%arg20: memref<768xf32>, %arg21: memref<768xf32>, %arg22: memref<f32>):
          %c0_85 = arith.constant 0 : index
          %c768_86 = arith.constant 768 : index
          %c1_87 = arith.constant 1 : index
          scf.for %arg23 = %c0_85 to %c768_86 step %c1_87 {
            %38 = memref.load %arg20[%arg23] : memref<768xf32>
            %39 = memref.load %arg21[%arg23] : memref<768xf32>
            %40 = memref.load %arg22[] : memref<f32>
            %41 = arith.mulf %38, %39 : f32
            %42 = arith.addf %41, %40 : f32
            memref.store %42, %arg22[] : memref<f32>
          }
        }
        cnm.gather %37[#map2] of %34 into %alloc_41 : !cnm.buffer<f32 on 1x6x8, level 0> into memref<48x1xf32>
        cnm.free_workgroup %34 : !cnm.workgroup<1x6x8>
        %alloc_83 = memref.alloc() : memref<288x1xf32>
        memref.copy %arg19, %alloc_83 : memref<288x1xf32> to memref<288x1xf32>
        %subview_84 = memref.subview %alloc_83[%arg18, 0] [48, 1] [1, 1] : memref<288x1xf32> to memref<48x1xf32, strided<[1, 1], offset: ?>>
        memref.copy %alloc_41, %subview_84 : memref<48x1xf32> to memref<48x1xf32, strided<[1, 1], offset: ?>>
        scf.yield %alloc_83 : memref<288x1xf32>
      }
      %reshape_78 = memref.reshape %28(%8) : (memref<288x1xf32>, memref<1xi64>) -> memref<288xf32>
      %29 = cnm.workgroup : !cnm.workgroup<1x6x8>
      %30 = cnm.alloc() for %29 : !cnm.buffer<6xf32 on 1x6x8, level 0>
      cnm.scatter %reshape_67 into %30[#map] of %29 : memref<48x6xf32> into !cnm.buffer<6xf32 on 1x6x8, level 0>
      %reshape_79 = memref.reshape %reshape_78(%17) : (memref<288xf32>, memref<2xi64>) -> memref<48x6xf32>
      %31 = cnm.alloc() for %29 : !cnm.buffer<6xf32 on 1x6x8, level 0>
      cnm.scatter %reshape_79 into %31[#map] of %29 : memref<48x6xf32> into !cnm.buffer<6xf32 on 1x6x8, level 0>
      %32 = cnm.alloc() for %29 : !cnm.buffer<6xf32 on 1x6x8, level 0>
      cnm.scatter %20 into %32[#map] of %29 : memref<48x6xf32> into !cnm.buffer<6xf32 on 1x6x8, level 0>
      cnm.launch %29 in(%30, %31 : !cnm.buffer<6xf32 on 1x6x8, level 0>, !cnm.buffer<6xf32 on 1x6x8, level 0>) out(%32 : !cnm.buffer<6xf32 on 1x6x8, level 0>) on !cnm.workgroup<1x6x8> {
      ^bb0(%arg18: memref<6xf32>, %arg19: memref<6xf32>, %arg20: memref<6xf32>):
        %c0_82 = arith.constant 0 : index
        %c6_83 = arith.constant 6 : index
        %c1_84 = arith.constant 1 : index
        scf.for %arg21 = %c0_82 to %c6_83 step %c1_84 {
          %33 = memref.load %arg18[%arg21] : memref<6xf32>
          %34 = memref.load %arg19[%arg21] : memref<6xf32>
          %35 = arith.addf %33, %34 : f32
          memref.store %35, %arg20[%arg21] : memref<6xf32>
        }
      }
      %alloc_80 = memref.alloc() : memref<48x6xf32>
      cnm.gather %32[#map] of %29 into %alloc_80 : !cnm.buffer<6xf32 on 1x6x8, level 0> into memref<48x6xf32>
      %reshape_81 = memref.reshape %alloc_80(%8) : (memref<48x6xf32>, memref<1xi64>) -> memref<288xf32>
      cnm.free_workgroup %29 : !cnm.workgroup<1x6x8>
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
      %6 = cnm.workgroup : !cnm.workgroup<2x8x16>
      scf.for %arg18 = %c0 to %c288 step %c1 {
        %10 = memref.load %reshape[%arg18, %c0] : memref<288x1xf32>
        memref.store %10, %alloc_48[%c0, %arg18] : memref<1x288xf32>
      }
      %7 = cnm.alloc() for %6 : !cnm.buffer<288xf32 on 2x8x16, level 0>
      %8 = cnm.alloc() for %6 : !cnm.buffer<288xf32 on 2x8x16, level 0>
      %9 = cnm.alloc() for %6 : !cnm.buffer<f32 on 2x8x16, level 0>
      cnm.scatter %subview_52 into %7[#map3] of %6 : memref<256x288xf32, strided<[288, 1], offset: ?>> into !cnm.buffer<288xf32 on 2x8x16, level 0>
      cnm.scatter %alloc_48 into %8[#map1] of %6 : memref<1x288xf32> into !cnm.buffer<288xf32 on 2x8x16, level 0>
      cnm.scatter %5 into %9[#map4] of %6 : memref<256x1xf32> into !cnm.buffer<f32 on 2x8x16, level 0>
      cnm.launch %6 in(%7, %8 : !cnm.buffer<288xf32 on 2x8x16, level 0>, !cnm.buffer<288xf32 on 2x8x16, level 0>) out(%9 : !cnm.buffer<f32 on 2x8x16, level 0>) on !cnm.workgroup<2x8x16> {
      ^bb0(%arg18: memref<288xf32>, %arg19: memref<288xf32>, %arg20: memref<f32>):
        %c0_55 = arith.constant 0 : index
        %c288_56 = arith.constant 288 : index
        %c1_57 = arith.constant 1 : index
        scf.for %arg21 = %c0_55 to %c288_56 step %c1_57 {
          %10 = memref.load %arg18[%arg21] : memref<288xf32>
          %11 = memref.load %arg19[%arg21] : memref<288xf32>
          %12 = memref.load %arg20[] : memref<f32>
          %13 = arith.mulf %10, %11 : f32
          %14 = arith.addf %13, %12 : f32
          memref.store %14, %arg20[] : memref<f32>
        }
      }
      cnm.gather %9[#map4] of %6 into %alloc_49 : !cnm.buffer<f32 on 2x8x16, level 0> into memref<256x1xf32>
      cnm.free_workgroup %6 : !cnm.workgroup<2x8x16>
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
        %6 = cnm.workgroup : !cnm.workgroup<1x1x8>
        %7 = memref.get_global @__constant_2xi64_3 : memref<2xi64>
        memref.copy %subview_18, %alloc_5 : memref<48xf32, strided<[1], offset: ?>> to memref<48xf32>
        %reshape = memref.reshape %alloc_5(%7) : (memref<48xf32>, memref<2xi64>) -> memref<8x6xf32>
        %8 = cnm.alloc() for %6 : !cnm.buffer<6xf32 on 1x1x8, level 0>
        cnm.scatter %reshape into %8[#map5] of %6 : memref<8x6xf32> into !cnm.buffer<6xf32 on 1x1x8, level 0>
        memref.copy %subview_19, %alloc_6 : memref<48xf32, strided<[1], offset: ?>> to memref<48xf32>
        %reshape_20 = memref.reshape %alloc_6(%7) : (memref<48xf32>, memref<2xi64>) -> memref<8x6xf32>
        %9 = cnm.alloc() for %6 : !cnm.buffer<6xf32 on 1x1x8, level 0>
        cnm.scatter %reshape_20 into %9[#map5] of %6 : memref<8x6xf32> into !cnm.buffer<6xf32 on 1x1x8, level 0>
        %10 = memref.get_global @__constant_8x6xf32 : memref<8x6xf32>
        %11 = cnm.alloc() for %6 : !cnm.buffer<6xf32 on 1x1x8, level 0>
        cnm.scatter %10 into %11[#map5] of %6 : memref<8x6xf32> into !cnm.buffer<6xf32 on 1x1x8, level 0>
        cnm.launch %6 in(%8, %9 : !cnm.buffer<6xf32 on 1x1x8, level 0>, !cnm.buffer<6xf32 on 1x1x8, level 0>) out(%11 : !cnm.buffer<6xf32 on 1x1x8, level 0>) on !cnm.workgroup<1x1x8> {
        ^bb0(%arg8: memref<6xf32>, %arg9: memref<6xf32>, %arg10: memref<6xf32>):
          %c0_23 = arith.constant 0 : index
          %c6_24 = arith.constant 6 : index
          %c1_25 = arith.constant 1 : index
          scf.for %arg11 = %c0_23 to %c6_24 step %c1_25 {
            %20 = memref.load %arg8[%arg11] : memref<6xf32>
            %21 = memref.load %arg9[%arg11] : memref<6xf32>
            %22 = arith.mulf %20, %21 : f32
            memref.store %22, %arg10[%arg11] : memref<6xf32>
          }
        }
        cnm.gather %11[#map5] of %6 into %alloc_7 : !cnm.buffer<6xf32 on 1x1x8, level 0> into memref<8x6xf32>
        %12 = memref.get_global @__constant_1xi64_3 : memref<1xi64>
        %reshape_21 = memref.reshape %alloc_7(%12) : (memref<8x6xf32>, memref<1xi64>) -> memref<48xf32>
        cnm.free_workgroup %6 : !cnm.workgroup<1x1x8>
        %13 = memref.get_global @__constant_xf32_0 : memref<f32>
        memref.copy %13, %alloc_9 : memref<f32> to memref<f32>
        scf.for %arg8 = %c0 to %c48 step %c1 {
          %20 = memref.load %reshape_21[%arg8] : memref<48xf32>
          %21 = memref.load %alloc_9[] : memref<f32>
          %22 = arith.addf %20, %21 : f32
          memref.store %22, %alloc_9[] : memref<f32>
        }
        %14 = memref.load %alloc_9[] : memref<f32>
        memref.store %14, %alloc_8[%c0] : memref<1xf32>
        memref.copy %13, %alloc_10 : memref<f32> to memref<f32>
        %15 = memref.load %alloc_8[%c0] : memref<1xf32>
        %16 = memref.load %alloc_10[] : memref<f32>
        %17 = arith.addf %15, %16 : f32
        memref.store %17, %alloc_10[] : memref<f32>
        %18 = memref.load %alloc_10[] : memref<f32>
        %19 = arith.divf %18, %cst_0 : f32
        %alloc_22 = memref.alloc() : memref<256xf32>
        memref.copy %arg7, %alloc_22 : memref<256xf32> to memref<256xf32>
        memref.store %19, %alloc_22[%arg6] : memref<256xf32>
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
        %7 = cnm.workgroup : !cnm.workgroup<1x1x8>
        %8 = memref.get_global @__constant_2xi64_3 : memref<2xi64>
        memref.copy %subview_18, %alloc_14 : memref<48xf32, strided<[1], offset: ?>> to memref<48xf32>
        %reshape = memref.reshape %alloc_14(%8) : (memref<48xf32>, memref<2xi64>) -> memref<8x6xf32>
        %9 = cnm.alloc() for %7 : !cnm.buffer<6xf32 on 1x1x8, level 0>
        cnm.scatter %reshape into %9[#map5] of %7 : memref<8x6xf32> into !cnm.buffer<6xf32 on 1x1x8, level 0>
        memref.store %6, %alloc_15[%c0] : memref<8xf32>
        memref.store %6, %alloc_15[%c1] : memref<8xf32>
        memref.store %6, %alloc_15[%c2] : memref<8xf32>
        memref.store %6, %alloc_15[%c3] : memref<8xf32>
        memref.store %6, %alloc_15[%c4] : memref<8xf32>
        memref.store %6, %alloc_15[%c5] : memref<8xf32>
        memref.store %6, %alloc_15[%c6] : memref<8xf32>
        memref.store %6, %alloc_15[%c7] : memref<8xf32>
        %10 = cnm.alloc() for %7 : !cnm.buffer<f32 on 1x1x8, level 0>
        cnm.scatter %alloc_15 into %10[#map1] of %7 : memref<8xf32> into !cnm.buffer<f32 on 1x1x8, level 0>
        %11 = memref.get_global @__constant_8x6xf32 : memref<8x6xf32>
        %12 = cnm.alloc() for %7 : !cnm.buffer<6xf32 on 1x1x8, level 0>
        cnm.scatter %11 into %12[#map5] of %7 : memref<8x6xf32> into !cnm.buffer<6xf32 on 1x1x8, level 0>
        cnm.launch %7 in(%9, %10 : !cnm.buffer<6xf32 on 1x1x8, level 0>, !cnm.buffer<f32 on 1x1x8, level 0>) out(%12 : !cnm.buffer<6xf32 on 1x1x8, level 0>) on !cnm.workgroup<1x1x8> {
        ^bb0(%arg8: memref<6xf32>, %arg9: memref<f32>, %arg10: memref<6xf32>):
          %c0_24 = arith.constant 0 : index
          %c6_25 = arith.constant 6 : index
          %c1_26 = arith.constant 1 : index
          scf.for %arg11 = %c0_24 to %c6_25 step %c1_26 {
            %18 = memref.load %arg8[%arg11] : memref<6xf32>
            %19 = memref.load %arg9[] : memref<f32>
            %20 = arith.mulf %18, %19 : f32
            memref.store %20, %arg10[%arg11] : memref<6xf32>
          }
        }
        cnm.gather %12[#map5] of %7 into %alloc_16 : !cnm.buffer<6xf32 on 1x1x8, level 0> into memref<8x6xf32>
        %13 = memref.get_global @__constant_1xi64_3 : memref<1xi64>
        %reshape_19 = memref.reshape %alloc_16(%13) : (memref<8x6xf32>, memref<1xi64>) -> memref<48xf32>
        cnm.free_workgroup %7 : !cnm.workgroup<1x1x8>
        %14 = cnm.workgroup : !cnm.workgroup<1x1x8>
        %reshape_20 = memref.reshape %arg7(%8) : (memref<48xf32>, memref<2xi64>) -> memref<8x6xf32>
        %15 = cnm.alloc() for %14 : !cnm.buffer<6xf32 on 1x1x8, level 0>
        cnm.scatter %reshape_20 into %15[#map5] of %14 : memref<8x6xf32> into !cnm.buffer<6xf32 on 1x1x8, level 0>
        %reshape_21 = memref.reshape %reshape_19(%8) : (memref<48xf32>, memref<2xi64>) -> memref<8x6xf32>
        %16 = cnm.alloc() for %14 : !cnm.buffer<6xf32 on 1x1x8, level 0>
        cnm.scatter %reshape_21 into %16[#map5] of %14 : memref<8x6xf32> into !cnm.buffer<6xf32 on 1x1x8, level 0>
        %17 = cnm.alloc() for %14 : !cnm.buffer<6xf32 on 1x1x8, level 0>
        cnm.scatter %11 into %17[#map5] of %14 : memref<8x6xf32> into !cnm.buffer<6xf32 on 1x1x8, level 0>
        cnm.launch %14 in(%15, %16 : !cnm.buffer<6xf32 on 1x1x8, level 0>, !cnm.buffer<6xf32 on 1x1x8, level 0>) out(%17 : !cnm.buffer<6xf32 on 1x1x8, level 0>) on !cnm.workgroup<1x1x8> {
        ^bb0(%arg8: memref<6xf32>, %arg9: memref<6xf32>, %arg10: memref<6xf32>):
          %c0_24 = arith.constant 0 : index
          %c6_25 = arith.constant 6 : index
          %c1_26 = arith.constant 1 : index
          scf.for %arg11 = %c0_24 to %c6_25 step %c1_26 {
            %18 = memref.load %arg8[%arg11] : memref<6xf32>
            %19 = memref.load %arg9[%arg11] : memref<6xf32>
            %20 = arith.addf %18, %19 : f32
            memref.store %20, %arg10[%arg11] : memref<6xf32>
          }
        }
        %alloc_22 = memref.alloc() : memref<8x6xf32>
        cnm.gather %17[#map5] of %14 into %alloc_22 : !cnm.buffer<6xf32 on 1x1x8, level 0> into memref<8x6xf32>
        %reshape_23 = memref.reshape %alloc_22(%13) : (memref<8x6xf32>, memref<1xi64>) -> memref<48xf32>
        cnm.free_workgroup %14 : !cnm.workgroup<1x1x8>
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
    %0 = cnm.workgroup : !cnm.workgroup<1x1x16>
    %1 = memref.get_global @__constant_2xi64_1 : memref<2xi64>
    %reshape = memref.reshape %arg0(%1) : (memref<288xf32>, memref<2xi64>) -> memref<16x18xf32>
    %2 = cnm.alloc() for %0 : !cnm.buffer<18xf32 on 1x1x16, level 0>
    cnm.scatter %reshape into %2[#map5] of %0 : memref<16x18xf32> into !cnm.buffer<18xf32 on 1x1x16, level 0>
    %3 = cnm.alloc() for %0 : !cnm.buffer<18xf32 on 1x1x16, level 0>
    cnm.scatter %reshape into %3[#map5] of %0 : memref<16x18xf32> into !cnm.buffer<18xf32 on 1x1x16, level 0>
    %4 = memref.get_global @__constant_16x18xf32 : memref<16x18xf32>
    %5 = cnm.alloc() for %0 : !cnm.buffer<18xf32 on 1x1x16, level 0>
    cnm.scatter %4 into %5[#map5] of %0 : memref<16x18xf32> into !cnm.buffer<18xf32 on 1x1x16, level 0>
    cnm.launch %0 in(%2, %3 : !cnm.buffer<18xf32 on 1x1x16, level 0>, !cnm.buffer<18xf32 on 1x1x16, level 0>) out(%5 : !cnm.buffer<18xf32 on 1x1x16, level 0>) on !cnm.workgroup<1x1x16> {
    ^bb0(%arg2: memref<18xf32>, %arg3: memref<18xf32>, %arg4: memref<18xf32>):
      %c0_12 = arith.constant 0 : index
      %c18 = arith.constant 18 : index
      %c1_13 = arith.constant 1 : index
      scf.for %arg5 = %c0_12 to %c18 step %c1_13 {
        %24 = memref.load %arg2[%arg5] : memref<18xf32>
        %25 = memref.load %arg3[%arg5] : memref<18xf32>
        %26 = arith.mulf %24, %25 : f32
        memref.store %26, %arg4[%arg5] : memref<18xf32>
      }
    }
    %alloc = memref.alloc() : memref<16x18xf32>
    cnm.gather %5[#map5] of %0 into %alloc : !cnm.buffer<18xf32 on 1x1x16, level 0> into memref<16x18xf32>
    %6 = memref.get_global @__constant_1xi64_1 : memref<1xi64>
    %reshape_1 = memref.reshape %alloc(%6) : (memref<16x18xf32>, memref<1xi64>) -> memref<288xf32>
    cnm.free_workgroup %0 : !cnm.workgroup<1x1x16>
    %7 = memref.get_global @__constant_xf32_0 : memref<f32>
    %alloc_2 = memref.alloc() : memref<1xf32>
    %alloc_3 = memref.alloc() : memref<f32>
    memref.copy %7, %alloc_3 : memref<f32> to memref<f32>
    scf.for %arg2 = %c0 to %c288 step %c1 {
      %24 = memref.load %reshape_1[%arg2] : memref<288xf32>
      %25 = memref.load %alloc_3[] : memref<f32>
      %26 = arith.addf %24, %25 : f32
      memref.store %26, %alloc_3[] : memref<f32>
    }
    %8 = memref.load %alloc_3[] : memref<f32>
    memref.store %8, %alloc_2[%c0] : memref<1xf32>
    %alloc_4 = memref.alloc() : memref<f32>
    memref.copy %7, %alloc_4 : memref<f32> to memref<f32>
    %9 = memref.load %alloc_2[%c0] : memref<1xf32>
    %10 = memref.load %alloc_4[] : memref<f32>
    %11 = arith.addf %9, %10 : f32
    memref.store %11, %alloc_4[] : memref<f32>
    %12 = memref.load %alloc_4[] : memref<f32>
    %13 = arith.divf %12, %cst_0 : f32
    %14 = arith.addf %13, %cst : f32
    %15 = math.rsqrt %14 : f32
    %16 = cnm.workgroup : !cnm.workgroup<1x1x16>
    %17 = cnm.alloc() for %16 : !cnm.buffer<18xf32 on 1x1x16, level 0>
    cnm.scatter %reshape into %17[#map5] of %16 : memref<16x18xf32> into !cnm.buffer<18xf32 on 1x1x16, level 0>
    %alloc_5 = memref.alloc() : memref<16xf32>
    memref.store %15, %alloc_5[%c0] : memref<16xf32>
    memref.store %15, %alloc_5[%c1] : memref<16xf32>
    memref.store %15, %alloc_5[%c2] : memref<16xf32>
    memref.store %15, %alloc_5[%c3] : memref<16xf32>
    memref.store %15, %alloc_5[%c4] : memref<16xf32>
    memref.store %15, %alloc_5[%c5] : memref<16xf32>
    memref.store %15, %alloc_5[%c6] : memref<16xf32>
    memref.store %15, %alloc_5[%c7] : memref<16xf32>
    memref.store %15, %alloc_5[%c8] : memref<16xf32>
    memref.store %15, %alloc_5[%c9] : memref<16xf32>
    memref.store %15, %alloc_5[%c10] : memref<16xf32>
    memref.store %15, %alloc_5[%c11] : memref<16xf32>
    memref.store %15, %alloc_5[%c12] : memref<16xf32>
    memref.store %15, %alloc_5[%c13] : memref<16xf32>
    memref.store %15, %alloc_5[%c14] : memref<16xf32>
    memref.store %15, %alloc_5[%c15] : memref<16xf32>
    %18 = cnm.alloc() for %16 : !cnm.buffer<f32 on 1x1x16, level 0>
    cnm.scatter %alloc_5 into %18[#map1] of %16 : memref<16xf32> into !cnm.buffer<f32 on 1x1x16, level 0>
    %19 = cnm.alloc() for %16 : !cnm.buffer<18xf32 on 1x1x16, level 0>
    cnm.scatter %4 into %19[#map5] of %16 : memref<16x18xf32> into !cnm.buffer<18xf32 on 1x1x16, level 0>
    cnm.launch %16 in(%17, %18 : !cnm.buffer<18xf32 on 1x1x16, level 0>, !cnm.buffer<f32 on 1x1x16, level 0>) out(%19 : !cnm.buffer<18xf32 on 1x1x16, level 0>) on !cnm.workgroup<1x1x16> {
    ^bb0(%arg2: memref<18xf32>, %arg3: memref<f32>, %arg4: memref<18xf32>):
      %c0_12 = arith.constant 0 : index
      %c18 = arith.constant 18 : index
      %c1_13 = arith.constant 1 : index
      scf.for %arg5 = %c0_12 to %c18 step %c1_13 {
        %24 = memref.load %arg2[%arg5] : memref<18xf32>
        %25 = memref.load %arg3[] : memref<f32>
        %26 = arith.mulf %24, %25 : f32
        memref.store %26, %arg4[%arg5] : memref<18xf32>
      }
    }
    %alloc_6 = memref.alloc() : memref<16x18xf32>
    cnm.gather %19[#map5] of %16 into %alloc_6 : !cnm.buffer<18xf32 on 1x1x16, level 0> into memref<16x18xf32>
    %reshape_7 = memref.reshape %alloc_6(%6) : (memref<16x18xf32>, memref<1xi64>) -> memref<288xf32>
    cnm.free_workgroup %16 : !cnm.workgroup<1x1x16>
    %20 = cnm.workgroup : !cnm.workgroup<1x1x16>
    %reshape_8 = memref.reshape %reshape_7(%1) : (memref<288xf32>, memref<2xi64>) -> memref<16x18xf32>
    %21 = cnm.alloc() for %20 : !cnm.buffer<18xf32 on 1x1x16, level 0>
    cnm.scatter %reshape_8 into %21[#map5] of %20 : memref<16x18xf32> into !cnm.buffer<18xf32 on 1x1x16, level 0>
    %reshape_9 = memref.reshape %arg1(%1) : (memref<288xf32>, memref<2xi64>) -> memref<16x18xf32>
    %22 = cnm.alloc() for %20 : !cnm.buffer<18xf32 on 1x1x16, level 0>
    cnm.scatter %reshape_9 into %22[#map5] of %20 : memref<16x18xf32> into !cnm.buffer<18xf32 on 1x1x16, level 0>
    %23 = cnm.alloc() for %20 : !cnm.buffer<18xf32 on 1x1x16, level 0>
    cnm.scatter %4 into %23[#map5] of %20 : memref<16x18xf32> into !cnm.buffer<18xf32 on 1x1x16, level 0>
    cnm.launch %20 in(%21, %22 : !cnm.buffer<18xf32 on 1x1x16, level 0>, !cnm.buffer<18xf32 on 1x1x16, level 0>) out(%23 : !cnm.buffer<18xf32 on 1x1x16, level 0>) on !cnm.workgroup<1x1x16> {
    ^bb0(%arg2: memref<18xf32>, %arg3: memref<18xf32>, %arg4: memref<18xf32>):
      %c0_12 = arith.constant 0 : index
      %c18 = arith.constant 18 : index
      %c1_13 = arith.constant 1 : index
      scf.for %arg5 = %c0_12 to %c18 step %c1_13 {
        %24 = memref.load %arg2[%arg5] : memref<18xf32>
        %25 = memref.load %arg3[%arg5] : memref<18xf32>
        %26 = arith.mulf %24, %25 : f32
        memref.store %26, %arg4[%arg5] : memref<18xf32>
      }
    }
    %alloc_10 = memref.alloc() : memref<16x18xf32>
    cnm.gather %23[#map5] of %20 into %alloc_10 : !cnm.buffer<18xf32 on 1x1x16, level 0> into memref<16x18xf32>
    %reshape_11 = memref.reshape %alloc_10(%6) : (memref<16x18xf32>, memref<1xi64>) -> memref<288xf32>
    cnm.free_workgroup %20 : !cnm.workgroup<1x1x16>
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
      %23 = memref.load %arg0[%arg1] : memref<256xf32>
      %24 = memref.load %alloc_0[] : memref<f32>
      %25 = arith.maximumf %23, %24 : f32
      memref.store %25, %alloc_0[] : memref<f32>
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
    %6 = cnm.workgroup : !cnm.workgroup<1x8x16>
    %7 = memref.get_global @__constant_2xi64_2 : memref<2xi64>
    %reshape = memref.reshape %arg0(%7) : (memref<256xf32>, memref<2xi64>) -> memref<128x2xf32>
    %8 = cnm.alloc() for %6 : !cnm.buffer<2xf32 on 1x8x16, level 0>
    cnm.scatter %reshape into %8[#map3] of %6 : memref<128x2xf32> into !cnm.buffer<2xf32 on 1x8x16, level 0>
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
    %9 = cnm.alloc() for %6 : !cnm.buffer<f32 on 1x8x16, level 0>
    cnm.scatter %alloc_2 into %9[#map1] of %6 : memref<16xf32> into !cnm.buffer<f32 on 1x8x16, level 0>
    %10 = memref.get_global @__constant_128x2xf32 : memref<128x2xf32>
    %11 = cnm.alloc() for %6 : !cnm.buffer<2xf32 on 1x8x16, level 0>
    cnm.scatter %10 into %11[#map3] of %6 : memref<128x2xf32> into !cnm.buffer<2xf32 on 1x8x16, level 0>
    cnm.launch %6 in(%8, %9 : !cnm.buffer<2xf32 on 1x8x16, level 0>, !cnm.buffer<f32 on 1x8x16, level 0>) out(%11 : !cnm.buffer<2xf32 on 1x8x16, level 0>) on !cnm.workgroup<1x8x16> {
    ^bb0(%arg1: memref<2xf32>, %arg2: memref<f32>, %arg3: memref<2xf32>):
      %c0_13 = arith.constant 0 : index
      %c2_14 = arith.constant 2 : index
      %c1_15 = arith.constant 1 : index
      scf.for %arg4 = %c0_13 to %c2_14 step %c1_15 {
        %23 = memref.load %arg1[%arg4] : memref<2xf32>
        %24 = memref.load %arg2[] : memref<f32>
        %25 = arith.subf %23, %24 : f32
        memref.store %25, %arg3[%arg4] : memref<2xf32>
      }
    }
    %alloc_3 = memref.alloc() : memref<128x2xf32>
    cnm.gather %11[#map3] of %6 into %alloc_3 : !cnm.buffer<2xf32 on 1x8x16, level 0> into memref<128x2xf32>
    %12 = memref.get_global @__constant_1xi64_2 : memref<1xi64>
    %reshape_4 = memref.reshape %alloc_3(%12) : (memref<128x2xf32>, memref<1xi64>) -> memref<256xf32>
    cnm.free_workgroup %6 : !cnm.workgroup<1x8x16>
    %alloc_5 = memref.alloc() : memref<256xf32>
    scf.for %arg1 = %c0 to %c256 step %c1 {
      %23 = memref.load %reshape_4[%arg1] : memref<256xf32>
      %24 = math.exp %23 : f32
      memref.store %24, %alloc_5[%arg1] : memref<256xf32>
    }
    %13 = memref.get_global @__constant_xf32_0 : memref<f32>
    %alloc_6 = memref.alloc() : memref<1xf32>
    %alloc_7 = memref.alloc() : memref<f32>
    memref.copy %13, %alloc_7 : memref<f32> to memref<f32>
    scf.for %arg1 = %c0 to %c256 step %c1 {
      %23 = memref.load %alloc_5[%arg1] : memref<256xf32>
      %24 = memref.load %alloc_7[] : memref<f32>
      %25 = arith.addf %23, %24 : f32
      memref.store %25, %alloc_7[] : memref<f32>
    }
    %14 = memref.load %alloc_7[] : memref<f32>
    memref.store %14, %alloc_6[%c0] : memref<1xf32>
    %alloc_8 = memref.alloc() : memref<f32>
    memref.copy %13, %alloc_8 : memref<f32> to memref<f32>
    %15 = memref.load %alloc_6[%c0] : memref<1xf32>
    %16 = memref.load %alloc_8[] : memref<f32>
    %17 = arith.addf %15, %16 : f32
    memref.store %17, %alloc_8[] : memref<f32>
    %18 = memref.load %alloc_8[] : memref<f32>
    %19 = cnm.workgroup : !cnm.workgroup<1x8x16>
    %reshape_9 = memref.reshape %alloc_5(%7) : (memref<256xf32>, memref<2xi64>) -> memref<128x2xf32>
    %20 = cnm.alloc() for %19 : !cnm.buffer<2xf32 on 1x8x16, level 0>
    cnm.scatter %reshape_9 into %20[#map3] of %19 : memref<128x2xf32> into !cnm.buffer<2xf32 on 1x8x16, level 0>
    %alloc_10 = memref.alloc() : memref<16xf32>
    memref.store %18, %alloc_10[%c0] : memref<16xf32>
    memref.store %18, %alloc_10[%c1] : memref<16xf32>
    memref.store %18, %alloc_10[%c2] : memref<16xf32>
    memref.store %18, %alloc_10[%c3] : memref<16xf32>
    memref.store %18, %alloc_10[%c4] : memref<16xf32>
    memref.store %18, %alloc_10[%c5] : memref<16xf32>
    memref.store %18, %alloc_10[%c6] : memref<16xf32>
    memref.store %18, %alloc_10[%c7] : memref<16xf32>
    memref.store %18, %alloc_10[%c8] : memref<16xf32>
    memref.store %18, %alloc_10[%c9] : memref<16xf32>
    memref.store %18, %alloc_10[%c10] : memref<16xf32>
    memref.store %18, %alloc_10[%c11] : memref<16xf32>
    memref.store %18, %alloc_10[%c12] : memref<16xf32>
    memref.store %18, %alloc_10[%c13] : memref<16xf32>
    memref.store %18, %alloc_10[%c14] : memref<16xf32>
    memref.store %18, %alloc_10[%c15] : memref<16xf32>
    %21 = cnm.alloc() for %19 : !cnm.buffer<f32 on 1x8x16, level 0>
    cnm.scatter %alloc_10 into %21[#map1] of %19 : memref<16xf32> into !cnm.buffer<f32 on 1x8x16, level 0>
    %22 = cnm.alloc() for %19 : !cnm.buffer<2xf32 on 1x8x16, level 0>
    cnm.scatter %10 into %22[#map3] of %19 : memref<128x2xf32> into !cnm.buffer<2xf32 on 1x8x16, level 0>
    cnm.launch %19 in(%20, %21 : !cnm.buffer<2xf32 on 1x8x16, level 0>, !cnm.buffer<f32 on 1x8x16, level 0>) out(%22 : !cnm.buffer<2xf32 on 1x8x16, level 0>) on !cnm.workgroup<1x8x16> {
    ^bb0(%arg1: memref<2xf32>, %arg2: memref<f32>, %arg3: memref<2xf32>):
      %c0_13 = arith.constant 0 : index
      %c2_14 = arith.constant 2 : index
      %c1_15 = arith.constant 1 : index
      scf.for %arg4 = %c0_13 to %c2_14 step %c1_15 {
        %23 = memref.load %arg1[%arg4] : memref<2xf32>
        %24 = memref.load %arg2[] : memref<f32>
        %25 = arith.divf %23, %24 : f32
        memref.store %25, %arg3[%arg4] : memref<2xf32>
      }
    }
    %alloc_11 = memref.alloc() : memref<128x2xf32>
    cnm.gather %22[#map3] of %19 into %alloc_11 : !cnm.buffer<2xf32 on 1x8x16, level 0> into memref<128x2xf32>
    %reshape_12 = memref.reshape %alloc_11(%12) : (memref<128x2xf32>, memref<1xi64>) -> memref<256xf32>
    cnm.free_workgroup %19 : !cnm.workgroup<1x8x16>
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
    %0 = cnm.workgroup : !cnm.workgroup<4x64x16>
    %1 = memref.get_global @__constant_2xi64 : memref<2xi64>
    %reshape = memref.reshape %arg0(%1) : (memref<1048576xf32>, memref<2xi64>) -> memref<4096x256xf32>
    %2 = cnm.alloc() for %0 : !cnm.buffer<256xf32 on 4x64x16, level 0>
    cnm.scatter %reshape into %2[#map6] of %0 : memref<4096x256xf32> into !cnm.buffer<256xf32 on 4x64x16, level 0>
    %3 = cnm.alloc() for %0 : !cnm.buffer<256xf32 on 4x64x16, level 0>
    cnm.scatter %reshape into %3[#map6] of %0 : memref<4096x256xf32> into !cnm.buffer<256xf32 on 4x64x16, level 0>
    %4 = memref.get_global @__constant_4096x256xf32 : memref<4096x256xf32>
    %5 = cnm.alloc() for %0 : !cnm.buffer<256xf32 on 4x64x16, level 0>
    cnm.scatter %4 into %5[#map6] of %0 : memref<4096x256xf32> into !cnm.buffer<256xf32 on 4x64x16, level 0>
    cnm.launch %0 in(%2, %3 : !cnm.buffer<256xf32 on 4x64x16, level 0>, !cnm.buffer<256xf32 on 4x64x16, level 0>) out(%5 : !cnm.buffer<256xf32 on 4x64x16, level 0>) on !cnm.workgroup<4x64x16> {
    ^bb0(%arg2: memref<256xf32>, %arg3: memref<256xf32>, %arg4: memref<256xf32>):
      %c0_12 = arith.constant 0 : index
      %c256 = arith.constant 256 : index
      %c1_13 = arith.constant 1 : index
      scf.for %arg5 = %c0_12 to %c256 step %c1_13 {
        %20 = memref.load %arg2[%arg5] : memref<256xf32>
        %21 = memref.load %arg3[%arg5] : memref<256xf32>
        %22 = arith.mulf %20, %21 : f32
        memref.store %22, %arg4[%arg5] : memref<256xf32>
      }
    }
    %alloc = memref.alloc() : memref<4096x256xf32>
    cnm.gather %5[#map6] of %0 into %alloc : !cnm.buffer<256xf32 on 4x64x16, level 0> into memref<4096x256xf32>
    %6 = memref.get_global @__constant_1xi64 : memref<1xi64>
    %reshape_1 = memref.reshape %alloc(%6) : (memref<4096x256xf32>, memref<1xi64>) -> memref<1048576xf32>
    cnm.free_workgroup %0 : !cnm.workgroup<4x64x16>
    %7 = memref.get_global @__constant_xf32_0 : memref<f32>
    %alloc_2 = memref.alloc() : memref<1024xf32>
    %alloc_3 = memref.alloc() : memref<f32>
    scf.for %arg2 = %c0 to %c1024 step %c1 {
      %20 = arith.muli %arg2, %c1024 : index
      %subview = memref.subview %reshape_1[%20] [1024] [1] : memref<1048576xf32> to memref<1024xf32, strided<[1], offset: ?>>
      memref.copy %7, %alloc_3 : memref<f32> to memref<f32>
      scf.for %arg3 = %c0 to %c1024 step %c1 {
        %22 = memref.load %subview[%arg3] : memref<1024xf32, strided<[1], offset: ?>>
        %23 = memref.load %alloc_3[] : memref<f32>
        %24 = arith.addf %22, %23 : f32
        memref.store %24, %alloc_3[] : memref<f32>
      }
      %21 = memref.load %alloc_3[] : memref<f32>
      memref.store %21, %alloc_2[%arg2] : memref<1024xf32>
    }
    %alloc_4 = memref.alloc() : memref<f32>
    memref.copy %7, %alloc_4 : memref<f32> to memref<f32>
    scf.for %arg2 = %c0 to %c1024 step %c1 {
      %20 = memref.load %alloc_2[%arg2] : memref<1024xf32>
      %21 = memref.load %alloc_4[] : memref<f32>
      %22 = arith.addf %20, %21 : f32
      memref.store %22, %alloc_4[] : memref<f32>
    }
    %8 = memref.load %alloc_4[] : memref<f32>
    %9 = arith.divf %8, %cst_0 : f32
    %10 = arith.addf %9, %cst : f32
    %11 = math.rsqrt %10 : f32
    %12 = cnm.workgroup : !cnm.workgroup<4x64x16>
    %13 = cnm.alloc() for %12 : !cnm.buffer<256xf32 on 4x64x16, level 0>
    cnm.scatter %reshape into %13[#map6] of %12 : memref<4096x256xf32> into !cnm.buffer<256xf32 on 4x64x16, level 0>
    %alloc_5 = memref.alloc() : memref<16xf32>
    memref.store %11, %alloc_5[%c0] : memref<16xf32>
    memref.store %11, %alloc_5[%c1] : memref<16xf32>
    memref.store %11, %alloc_5[%c2] : memref<16xf32>
    memref.store %11, %alloc_5[%c3] : memref<16xf32>
    memref.store %11, %alloc_5[%c4] : memref<16xf32>
    memref.store %11, %alloc_5[%c5] : memref<16xf32>
    memref.store %11, %alloc_5[%c6] : memref<16xf32>
    memref.store %11, %alloc_5[%c7] : memref<16xf32>
    memref.store %11, %alloc_5[%c8] : memref<16xf32>
    memref.store %11, %alloc_5[%c9] : memref<16xf32>
    memref.store %11, %alloc_5[%c10] : memref<16xf32>
    memref.store %11, %alloc_5[%c11] : memref<16xf32>
    memref.store %11, %alloc_5[%c12] : memref<16xf32>
    memref.store %11, %alloc_5[%c13] : memref<16xf32>
    memref.store %11, %alloc_5[%c14] : memref<16xf32>
    memref.store %11, %alloc_5[%c15] : memref<16xf32>
    %14 = cnm.alloc() for %12 : !cnm.buffer<f32 on 4x64x16, level 0>
    cnm.scatter %alloc_5 into %14[#map1] of %12 : memref<16xf32> into !cnm.buffer<f32 on 4x64x16, level 0>
    %15 = cnm.alloc() for %12 : !cnm.buffer<256xf32 on 4x64x16, level 0>
    cnm.scatter %4 into %15[#map6] of %12 : memref<4096x256xf32> into !cnm.buffer<256xf32 on 4x64x16, level 0>
    cnm.launch %12 in(%13, %14 : !cnm.buffer<256xf32 on 4x64x16, level 0>, !cnm.buffer<f32 on 4x64x16, level 0>) out(%15 : !cnm.buffer<256xf32 on 4x64x16, level 0>) on !cnm.workgroup<4x64x16> {
    ^bb0(%arg2: memref<256xf32>, %arg3: memref<f32>, %arg4: memref<256xf32>):
      %c0_12 = arith.constant 0 : index
      %c256 = arith.constant 256 : index
      %c1_13 = arith.constant 1 : index
      scf.for %arg5 = %c0_12 to %c256 step %c1_13 {
        %20 = memref.load %arg2[%arg5] : memref<256xf32>
        %21 = memref.load %arg3[] : memref<f32>
        %22 = arith.mulf %20, %21 : f32
        memref.store %22, %arg4[%arg5] : memref<256xf32>
      }
    }
    %alloc_6 = memref.alloc() : memref<4096x256xf32>
    cnm.gather %15[#map6] of %12 into %alloc_6 : !cnm.buffer<256xf32 on 4x64x16, level 0> into memref<4096x256xf32>
    %reshape_7 = memref.reshape %alloc_6(%6) : (memref<4096x256xf32>, memref<1xi64>) -> memref<1048576xf32>
    cnm.free_workgroup %12 : !cnm.workgroup<4x64x16>
    %16 = cnm.workgroup : !cnm.workgroup<4x64x16>
    %reshape_8 = memref.reshape %reshape_7(%1) : (memref<1048576xf32>, memref<2xi64>) -> memref<4096x256xf32>
    %17 = cnm.alloc() for %16 : !cnm.buffer<256xf32 on 4x64x16, level 0>
    cnm.scatter %reshape_8 into %17[#map6] of %16 : memref<4096x256xf32> into !cnm.buffer<256xf32 on 4x64x16, level 0>
    %reshape_9 = memref.reshape %arg1(%1) : (memref<1048576xf32>, memref<2xi64>) -> memref<4096x256xf32>
    %18 = cnm.alloc() for %16 : !cnm.buffer<256xf32 on 4x64x16, level 0>
    cnm.scatter %reshape_9 into %18[#map6] of %16 : memref<4096x256xf32> into !cnm.buffer<256xf32 on 4x64x16, level 0>
    %19 = cnm.alloc() for %16 : !cnm.buffer<256xf32 on 4x64x16, level 0>
    cnm.scatter %4 into %19[#map6] of %16 : memref<4096x256xf32> into !cnm.buffer<256xf32 on 4x64x16, level 0>
    cnm.launch %16 in(%17, %18 : !cnm.buffer<256xf32 on 4x64x16, level 0>, !cnm.buffer<256xf32 on 4x64x16, level 0>) out(%19 : !cnm.buffer<256xf32 on 4x64x16, level 0>) on !cnm.workgroup<4x64x16> {
    ^bb0(%arg2: memref<256xf32>, %arg3: memref<256xf32>, %arg4: memref<256xf32>):
      %c0_12 = arith.constant 0 : index
      %c256 = arith.constant 256 : index
      %c1_13 = arith.constant 1 : index
      scf.for %arg5 = %c0_12 to %c256 step %c1_13 {
        %20 = memref.load %arg2[%arg5] : memref<256xf32>
        %21 = memref.load %arg3[%arg5] : memref<256xf32>
        %22 = arith.mulf %20, %21 : f32
        memref.store %22, %arg4[%arg5] : memref<256xf32>
      }
    }
    %alloc_10 = memref.alloc() : memref<4096x256xf32>
    cnm.gather %19[#map6] of %16 into %alloc_10 : !cnm.buffer<256xf32 on 4x64x16, level 0> into memref<4096x256xf32>
    %reshape_11 = memref.reshape %alloc_10(%6) : (memref<4096x256xf32>, memref<1xi64>) -> memref<1048576xf32>
    cnm.free_workgroup %16 : !cnm.workgroup<4x64x16>
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
      %15 = arith.muli %arg1, %c1024 : index
      %subview = memref.subview %arg0[%15] [1024] [1] : memref<1048576xf32> to memref<1024xf32, strided<[1], offset: ?>>
      memref.copy %0, %alloc_0 : memref<f32> to memref<f32>
      scf.for %arg2 = %c0 to %c1024 step %c1 {
        %17 = memref.load %subview[%arg2] : memref<1024xf32, strided<[1], offset: ?>>
        %18 = memref.load %alloc_0[] : memref<f32>
        %19 = arith.maximumf %17, %18 : f32
        memref.store %19, %alloc_0[] : memref<f32>
      }
      %16 = memref.load %alloc_0[] : memref<f32>
      memref.store %16, %alloc[%arg1] : memref<1024xf32>
    }
    %alloc_1 = memref.alloc() : memref<f32>
    memref.copy %0, %alloc_1 : memref<f32> to memref<f32>
    scf.for %arg1 = %c0 to %c1024 step %c1 {
      %15 = memref.load %alloc[%arg1] : memref<1024xf32>
      %16 = memref.load %alloc_1[] : memref<f32>
      %17 = arith.maximumf %15, %16 : f32
      memref.store %17, %alloc_1[] : memref<f32>
    }
    %1 = memref.load %alloc_1[] : memref<f32>
    %2 = cnm.workgroup : !cnm.workgroup<4x64x16>
    %3 = memref.get_global @__constant_2xi64 : memref<2xi64>
    %reshape = memref.reshape %arg0(%3) : (memref<1048576xf32>, memref<2xi64>) -> memref<4096x256xf32>
    %4 = cnm.alloc() for %2 : !cnm.buffer<256xf32 on 4x64x16, level 0>
    cnm.scatter %reshape into %4[#map6] of %2 : memref<4096x256xf32> into !cnm.buffer<256xf32 on 4x64x16, level 0>
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
    %5 = cnm.alloc() for %2 : !cnm.buffer<f32 on 4x64x16, level 0>
    cnm.scatter %alloc_2 into %5[#map1] of %2 : memref<16xf32> into !cnm.buffer<f32 on 4x64x16, level 0>
    %6 = memref.get_global @__constant_4096x256xf32 : memref<4096x256xf32>
    %7 = cnm.alloc() for %2 : !cnm.buffer<256xf32 on 4x64x16, level 0>
    cnm.scatter %6 into %7[#map6] of %2 : memref<4096x256xf32> into !cnm.buffer<256xf32 on 4x64x16, level 0>
    cnm.launch %2 in(%4, %5 : !cnm.buffer<256xf32 on 4x64x16, level 0>, !cnm.buffer<f32 on 4x64x16, level 0>) out(%7 : !cnm.buffer<256xf32 on 4x64x16, level 0>) on !cnm.workgroup<4x64x16> {
    ^bb0(%arg1: memref<256xf32>, %arg2: memref<f32>, %arg3: memref<256xf32>):
      %c0_13 = arith.constant 0 : index
      %c256 = arith.constant 256 : index
      %c1_14 = arith.constant 1 : index
      scf.for %arg4 = %c0_13 to %c256 step %c1_14 {
        %15 = memref.load %arg1[%arg4] : memref<256xf32>
        %16 = memref.load %arg2[] : memref<f32>
        %17 = arith.subf %15, %16 : f32
        memref.store %17, %arg3[%arg4] : memref<256xf32>
      }
    }
    %alloc_3 = memref.alloc() : memref<4096x256xf32>
    cnm.gather %7[#map6] of %2 into %alloc_3 : !cnm.buffer<256xf32 on 4x64x16, level 0> into memref<4096x256xf32>
    %8 = memref.get_global @__constant_1xi64 : memref<1xi64>
    %reshape_4 = memref.reshape %alloc_3(%8) : (memref<4096x256xf32>, memref<1xi64>) -> memref<1048576xf32>
    cnm.free_workgroup %2 : !cnm.workgroup<4x64x16>
    %alloc_5 = memref.alloc() : memref<1048576xf32>
    scf.for %arg1 = %c0 to %c1048576 step %c1 {
      %15 = memref.load %reshape_4[%arg1] : memref<1048576xf32>
      %16 = math.exp %15 : f32
      memref.store %16, %alloc_5[%arg1] : memref<1048576xf32>
    }
    %9 = memref.get_global @__constant_xf32_0 : memref<f32>
    %alloc_6 = memref.alloc() : memref<1024xf32>
    %alloc_7 = memref.alloc() : memref<f32>
    scf.for %arg1 = %c0 to %c1024 step %c1 {
      %15 = arith.muli %arg1, %c1024 : index
      %subview = memref.subview %alloc_5[%15] [1024] [1] : memref<1048576xf32> to memref<1024xf32, strided<[1], offset: ?>>
      memref.copy %9, %alloc_7 : memref<f32> to memref<f32>
      scf.for %arg2 = %c0 to %c1024 step %c1 {
        %17 = memref.load %subview[%arg2] : memref<1024xf32, strided<[1], offset: ?>>
        %18 = memref.load %alloc_7[] : memref<f32>
        %19 = arith.addf %17, %18 : f32
        memref.store %19, %alloc_7[] : memref<f32>
      }
      %16 = memref.load %alloc_7[] : memref<f32>
      memref.store %16, %alloc_6[%arg1] : memref<1024xf32>
    }
    %alloc_8 = memref.alloc() : memref<f32>
    memref.copy %9, %alloc_8 : memref<f32> to memref<f32>
    scf.for %arg1 = %c0 to %c1024 step %c1 {
      %15 = memref.load %alloc_6[%arg1] : memref<1024xf32>
      %16 = memref.load %alloc_8[] : memref<f32>
      %17 = arith.addf %15, %16 : f32
      memref.store %17, %alloc_8[] : memref<f32>
    }
    %10 = memref.load %alloc_8[] : memref<f32>
    %11 = cnm.workgroup : !cnm.workgroup<4x64x16>
    %reshape_9 = memref.reshape %alloc_5(%3) : (memref<1048576xf32>, memref<2xi64>) -> memref<4096x256xf32>
    %12 = cnm.alloc() for %11 : !cnm.buffer<256xf32 on 4x64x16, level 0>
    cnm.scatter %reshape_9 into %12[#map6] of %11 : memref<4096x256xf32> into !cnm.buffer<256xf32 on 4x64x16, level 0>
    %alloc_10 = memref.alloc() : memref<16xf32>
    memref.store %10, %alloc_10[%c0] : memref<16xf32>
    memref.store %10, %alloc_10[%c1] : memref<16xf32>
    memref.store %10, %alloc_10[%c2] : memref<16xf32>
    memref.store %10, %alloc_10[%c3] : memref<16xf32>
    memref.store %10, %alloc_10[%c4] : memref<16xf32>
    memref.store %10, %alloc_10[%c5] : memref<16xf32>
    memref.store %10, %alloc_10[%c6] : memref<16xf32>
    memref.store %10, %alloc_10[%c7] : memref<16xf32>
    memref.store %10, %alloc_10[%c8] : memref<16xf32>
    memref.store %10, %alloc_10[%c9] : memref<16xf32>
    memref.store %10, %alloc_10[%c10] : memref<16xf32>
    memref.store %10, %alloc_10[%c11] : memref<16xf32>
    memref.store %10, %alloc_10[%c12] : memref<16xf32>
    memref.store %10, %alloc_10[%c13] : memref<16xf32>
    memref.store %10, %alloc_10[%c14] : memref<16xf32>
    memref.store %10, %alloc_10[%c15] : memref<16xf32>
    %13 = cnm.alloc() for %11 : !cnm.buffer<f32 on 4x64x16, level 0>
    cnm.scatter %alloc_10 into %13[#map1] of %11 : memref<16xf32> into !cnm.buffer<f32 on 4x64x16, level 0>
    %14 = cnm.alloc() for %11 : !cnm.buffer<256xf32 on 4x64x16, level 0>
    cnm.scatter %6 into %14[#map6] of %11 : memref<4096x256xf32> into !cnm.buffer<256xf32 on 4x64x16, level 0>
    cnm.launch %11 in(%12, %13 : !cnm.buffer<256xf32 on 4x64x16, level 0>, !cnm.buffer<f32 on 4x64x16, level 0>) out(%14 : !cnm.buffer<256xf32 on 4x64x16, level 0>) on !cnm.workgroup<4x64x16> {
    ^bb0(%arg1: memref<256xf32>, %arg2: memref<f32>, %arg3: memref<256xf32>):
      %c0_13 = arith.constant 0 : index
      %c256 = arith.constant 256 : index
      %c1_14 = arith.constant 1 : index
      scf.for %arg4 = %c0_13 to %c256 step %c1_14 {
        %15 = memref.load %arg1[%arg4] : memref<256xf32>
        %16 = memref.load %arg2[] : memref<f32>
        %17 = arith.divf %15, %16 : f32
        memref.store %17, %arg3[%arg4] : memref<256xf32>
      }
    }
    %alloc_11 = memref.alloc() : memref<4096x256xf32>
    cnm.gather %14[#map6] of %11 into %alloc_11 : !cnm.buffer<256xf32 on 4x64x16, level 0> into memref<4096x256xf32>
    %reshape_12 = memref.reshape %alloc_11(%8) : (memref<4096x256xf32>, memref<1xi64>) -> memref<1048576xf32>
    cnm.free_workgroup %11 : !cnm.workgroup<4x64x16>
    return %reshape_12 : memref<1048576xf32>
  }
  func.func @va_1048576(%arg0: memref<1048576xf32>, %arg1: memref<1048576xf32>) -> memref<1048576xf32> {
    %0 = cnm.workgroup : !cnm.workgroup<4x64x16>
    %1 = memref.get_global @__constant_2xi64 : memref<2xi64>
    %reshape = memref.reshape %arg0(%1) : (memref<1048576xf32>, memref<2xi64>) -> memref<4096x256xf32>
    %2 = cnm.alloc() for %0 : !cnm.buffer<256xf32 on 4x64x16, level 0>
    cnm.scatter %reshape into %2[#map6] of %0 : memref<4096x256xf32> into !cnm.buffer<256xf32 on 4x64x16, level 0>
    %reshape_0 = memref.reshape %arg1(%1) : (memref<1048576xf32>, memref<2xi64>) -> memref<4096x256xf32>
    %3 = cnm.alloc() for %0 : !cnm.buffer<256xf32 on 4x64x16, level 0>
    cnm.scatter %reshape_0 into %3[#map6] of %0 : memref<4096x256xf32> into !cnm.buffer<256xf32 on 4x64x16, level 0>
    %4 = memref.get_global @__constant_4096x256xf32 : memref<4096x256xf32>
    %5 = cnm.alloc() for %0 : !cnm.buffer<256xf32 on 4x64x16, level 0>
    cnm.scatter %4 into %5[#map6] of %0 : memref<4096x256xf32> into !cnm.buffer<256xf32 on 4x64x16, level 0>
    cnm.launch %0 in(%2, %3 : !cnm.buffer<256xf32 on 4x64x16, level 0>, !cnm.buffer<256xf32 on 4x64x16, level 0>) out(%5 : !cnm.buffer<256xf32 on 4x64x16, level 0>) on !cnm.workgroup<4x64x16> {
    ^bb0(%arg2: memref<256xf32>, %arg3: memref<256xf32>, %arg4: memref<256xf32>):
      %c0 = arith.constant 0 : index
      %c256 = arith.constant 256 : index
      %c1 = arith.constant 1 : index
      scf.for %arg5 = %c0 to %c256 step %c1 {
        %7 = memref.load %arg2[%arg5] : memref<256xf32>
        %8 = memref.load %arg3[%arg5] : memref<256xf32>
        %9 = arith.addf %7, %8 : f32
        memref.store %9, %arg4[%arg5] : memref<256xf32>
      }
    }
    %alloc = memref.alloc() : memref<4096x256xf32>
    cnm.gather %5[#map6] of %0 into %alloc : !cnm.buffer<256xf32 on 4x64x16, level 0> into memref<4096x256xf32>
    %6 = memref.get_global @__constant_1xi64 : memref<1xi64>
    %reshape_1 = memref.reshape %alloc(%6) : (memref<4096x256xf32>, memref<1xi64>) -> memref<1048576xf32>
    cnm.free_workgroup %0 : !cnm.workgroup<4x64x16>
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
    %0 = cnm.workgroup : !cnm.workgroup<4x64x16>
    %1 = memref.get_global @__constant_2xi64_0 : memref<2xi64>
    %reshape = memref.reshape %arg0(%1) : (memref<262144xf32>, memref<2xi64>) -> memref<4096x64xf32>
    %2 = cnm.alloc() for %0 : !cnm.buffer<64xf32 on 4x64x16, level 0>
    cnm.scatter %reshape into %2[#map6] of %0 : memref<4096x64xf32> into !cnm.buffer<64xf32 on 4x64x16, level 0>
    %3 = cnm.alloc() for %0 : !cnm.buffer<64xf32 on 4x64x16, level 0>
    cnm.scatter %reshape into %3[#map6] of %0 : memref<4096x64xf32> into !cnm.buffer<64xf32 on 4x64x16, level 0>
    %4 = memref.get_global @__constant_4096x64xf32 : memref<4096x64xf32>
    %5 = cnm.alloc() for %0 : !cnm.buffer<64xf32 on 4x64x16, level 0>
    cnm.scatter %4 into %5[#map6] of %0 : memref<4096x64xf32> into !cnm.buffer<64xf32 on 4x64x16, level 0>
    cnm.launch %0 in(%2, %3 : !cnm.buffer<64xf32 on 4x64x16, level 0>, !cnm.buffer<64xf32 on 4x64x16, level 0>) out(%5 : !cnm.buffer<64xf32 on 4x64x16, level 0>) on !cnm.workgroup<4x64x16> {
    ^bb0(%arg2: memref<64xf32>, %arg3: memref<64xf32>, %arg4: memref<64xf32>):
      %c0_12 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c1_13 = arith.constant 1 : index
      scf.for %arg5 = %c0_12 to %c64 step %c1_13 {
        %20 = memref.load %arg2[%arg5] : memref<64xf32>
        %21 = memref.load %arg3[%arg5] : memref<64xf32>
        %22 = arith.mulf %20, %21 : f32
        memref.store %22, %arg4[%arg5] : memref<64xf32>
      }
    }
    %alloc = memref.alloc() : memref<4096x64xf32>
    cnm.gather %5[#map6] of %0 into %alloc : !cnm.buffer<64xf32 on 4x64x16, level 0> into memref<4096x64xf32>
    %6 = memref.get_global @__constant_1xi64_0 : memref<1xi64>
    %reshape_1 = memref.reshape %alloc(%6) : (memref<4096x64xf32>, memref<1xi64>) -> memref<262144xf32>
    cnm.free_workgroup %0 : !cnm.workgroup<4x64x16>
    %7 = memref.get_global @__constant_xf32_0 : memref<f32>
    %alloc_2 = memref.alloc() : memref<256xf32>
    %alloc_3 = memref.alloc() : memref<f32>
    scf.for %arg2 = %c0 to %c256 step %c1 {
      %20 = arith.muli %arg2, %c1024 : index
      %subview = memref.subview %reshape_1[%20] [1024] [1] : memref<262144xf32> to memref<1024xf32, strided<[1], offset: ?>>
      memref.copy %7, %alloc_3 : memref<f32> to memref<f32>
      scf.for %arg3 = %c0 to %c1024 step %c1 {
        %22 = memref.load %subview[%arg3] : memref<1024xf32, strided<[1], offset: ?>>
        %23 = memref.load %alloc_3[] : memref<f32>
        %24 = arith.addf %22, %23 : f32
        memref.store %24, %alloc_3[] : memref<f32>
      }
      %21 = memref.load %alloc_3[] : memref<f32>
      memref.store %21, %alloc_2[%arg2] : memref<256xf32>
    }
    %alloc_4 = memref.alloc() : memref<f32>
    memref.copy %7, %alloc_4 : memref<f32> to memref<f32>
    scf.for %arg2 = %c0 to %c256 step %c1 {
      %20 = memref.load %alloc_2[%arg2] : memref<256xf32>
      %21 = memref.load %alloc_4[] : memref<f32>
      %22 = arith.addf %20, %21 : f32
      memref.store %22, %alloc_4[] : memref<f32>
    }
    %8 = memref.load %alloc_4[] : memref<f32>
    %9 = arith.divf %8, %cst_0 : f32
    %10 = arith.addf %9, %cst : f32
    %11 = math.rsqrt %10 : f32
    %12 = cnm.workgroup : !cnm.workgroup<4x64x16>
    %13 = cnm.alloc() for %12 : !cnm.buffer<64xf32 on 4x64x16, level 0>
    cnm.scatter %reshape into %13[#map6] of %12 : memref<4096x64xf32> into !cnm.buffer<64xf32 on 4x64x16, level 0>
    %alloc_5 = memref.alloc() : memref<16xf32>
    memref.store %11, %alloc_5[%c0] : memref<16xf32>
    memref.store %11, %alloc_5[%c1] : memref<16xf32>
    memref.store %11, %alloc_5[%c2] : memref<16xf32>
    memref.store %11, %alloc_5[%c3] : memref<16xf32>
    memref.store %11, %alloc_5[%c4] : memref<16xf32>
    memref.store %11, %alloc_5[%c5] : memref<16xf32>
    memref.store %11, %alloc_5[%c6] : memref<16xf32>
    memref.store %11, %alloc_5[%c7] : memref<16xf32>
    memref.store %11, %alloc_5[%c8] : memref<16xf32>
    memref.store %11, %alloc_5[%c9] : memref<16xf32>
    memref.store %11, %alloc_5[%c10] : memref<16xf32>
    memref.store %11, %alloc_5[%c11] : memref<16xf32>
    memref.store %11, %alloc_5[%c12] : memref<16xf32>
    memref.store %11, %alloc_5[%c13] : memref<16xf32>
    memref.store %11, %alloc_5[%c14] : memref<16xf32>
    memref.store %11, %alloc_5[%c15] : memref<16xf32>
    %14 = cnm.alloc() for %12 : !cnm.buffer<f32 on 4x64x16, level 0>
    cnm.scatter %alloc_5 into %14[#map1] of %12 : memref<16xf32> into !cnm.buffer<f32 on 4x64x16, level 0>
    %15 = cnm.alloc() for %12 : !cnm.buffer<64xf32 on 4x64x16, level 0>
    cnm.scatter %4 into %15[#map6] of %12 : memref<4096x64xf32> into !cnm.buffer<64xf32 on 4x64x16, level 0>
    cnm.launch %12 in(%13, %14 : !cnm.buffer<64xf32 on 4x64x16, level 0>, !cnm.buffer<f32 on 4x64x16, level 0>) out(%15 : !cnm.buffer<64xf32 on 4x64x16, level 0>) on !cnm.workgroup<4x64x16> {
    ^bb0(%arg2: memref<64xf32>, %arg3: memref<f32>, %arg4: memref<64xf32>):
      %c0_12 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c1_13 = arith.constant 1 : index
      scf.for %arg5 = %c0_12 to %c64 step %c1_13 {
        %20 = memref.load %arg2[%arg5] : memref<64xf32>
        %21 = memref.load %arg3[] : memref<f32>
        %22 = arith.mulf %20, %21 : f32
        memref.store %22, %arg4[%arg5] : memref<64xf32>
      }
    }
    %alloc_6 = memref.alloc() : memref<4096x64xf32>
    cnm.gather %15[#map6] of %12 into %alloc_6 : !cnm.buffer<64xf32 on 4x64x16, level 0> into memref<4096x64xf32>
    %reshape_7 = memref.reshape %alloc_6(%6) : (memref<4096x64xf32>, memref<1xi64>) -> memref<262144xf32>
    cnm.free_workgroup %12 : !cnm.workgroup<4x64x16>
    %16 = cnm.workgroup : !cnm.workgroup<4x64x16>
    %reshape_8 = memref.reshape %reshape_7(%1) : (memref<262144xf32>, memref<2xi64>) -> memref<4096x64xf32>
    %17 = cnm.alloc() for %16 : !cnm.buffer<64xf32 on 4x64x16, level 0>
    cnm.scatter %reshape_8 into %17[#map6] of %16 : memref<4096x64xf32> into !cnm.buffer<64xf32 on 4x64x16, level 0>
    %reshape_9 = memref.reshape %arg1(%1) : (memref<262144xf32>, memref<2xi64>) -> memref<4096x64xf32>
    %18 = cnm.alloc() for %16 : !cnm.buffer<64xf32 on 4x64x16, level 0>
    cnm.scatter %reshape_9 into %18[#map6] of %16 : memref<4096x64xf32> into !cnm.buffer<64xf32 on 4x64x16, level 0>
    %19 = cnm.alloc() for %16 : !cnm.buffer<64xf32 on 4x64x16, level 0>
    cnm.scatter %4 into %19[#map6] of %16 : memref<4096x64xf32> into !cnm.buffer<64xf32 on 4x64x16, level 0>
    cnm.launch %16 in(%17, %18 : !cnm.buffer<64xf32 on 4x64x16, level 0>, !cnm.buffer<64xf32 on 4x64x16, level 0>) out(%19 : !cnm.buffer<64xf32 on 4x64x16, level 0>) on !cnm.workgroup<4x64x16> {
    ^bb0(%arg2: memref<64xf32>, %arg3: memref<64xf32>, %arg4: memref<64xf32>):
      %c0_12 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c1_13 = arith.constant 1 : index
      scf.for %arg5 = %c0_12 to %c64 step %c1_13 {
        %20 = memref.load %arg2[%arg5] : memref<64xf32>
        %21 = memref.load %arg3[%arg5] : memref<64xf32>
        %22 = arith.mulf %20, %21 : f32
        memref.store %22, %arg4[%arg5] : memref<64xf32>
      }
    }
    %alloc_10 = memref.alloc() : memref<4096x64xf32>
    cnm.gather %19[#map6] of %16 into %alloc_10 : !cnm.buffer<64xf32 on 4x64x16, level 0> into memref<4096x64xf32>
    %reshape_11 = memref.reshape %alloc_10(%6) : (memref<4096x64xf32>, memref<1xi64>) -> memref<262144xf32>
    cnm.free_workgroup %16 : !cnm.workgroup<4x64x16>
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
      %15 = arith.muli %arg1, %c1024 : index
      %subview = memref.subview %arg0[%15] [1024] [1] : memref<262144xf32> to memref<1024xf32, strided<[1], offset: ?>>
      memref.copy %0, %alloc_0 : memref<f32> to memref<f32>
      scf.for %arg2 = %c0 to %c1024 step %c1 {
        %17 = memref.load %subview[%arg2] : memref<1024xf32, strided<[1], offset: ?>>
        %18 = memref.load %alloc_0[] : memref<f32>
        %19 = arith.maximumf %17, %18 : f32
        memref.store %19, %alloc_0[] : memref<f32>
      }
      %16 = memref.load %alloc_0[] : memref<f32>
      memref.store %16, %alloc[%arg1] : memref<256xf32>
    }
    %alloc_1 = memref.alloc() : memref<f32>
    memref.copy %0, %alloc_1 : memref<f32> to memref<f32>
    scf.for %arg1 = %c0 to %c256 step %c1 {
      %15 = memref.load %alloc[%arg1] : memref<256xf32>
      %16 = memref.load %alloc_1[] : memref<f32>
      %17 = arith.maximumf %15, %16 : f32
      memref.store %17, %alloc_1[] : memref<f32>
    }
    %1 = memref.load %alloc_1[] : memref<f32>
    %2 = cnm.workgroup : !cnm.workgroup<4x64x16>
    %3 = memref.get_global @__constant_2xi64_0 : memref<2xi64>
    %reshape = memref.reshape %arg0(%3) : (memref<262144xf32>, memref<2xi64>) -> memref<4096x64xf32>
    %4 = cnm.alloc() for %2 : !cnm.buffer<64xf32 on 4x64x16, level 0>
    cnm.scatter %reshape into %4[#map6] of %2 : memref<4096x64xf32> into !cnm.buffer<64xf32 on 4x64x16, level 0>
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
    %5 = cnm.alloc() for %2 : !cnm.buffer<f32 on 4x64x16, level 0>
    cnm.scatter %alloc_2 into %5[#map1] of %2 : memref<16xf32> into !cnm.buffer<f32 on 4x64x16, level 0>
    %6 = memref.get_global @__constant_4096x64xf32 : memref<4096x64xf32>
    %7 = cnm.alloc() for %2 : !cnm.buffer<64xf32 on 4x64x16, level 0>
    cnm.scatter %6 into %7[#map6] of %2 : memref<4096x64xf32> into !cnm.buffer<64xf32 on 4x64x16, level 0>
    cnm.launch %2 in(%4, %5 : !cnm.buffer<64xf32 on 4x64x16, level 0>, !cnm.buffer<f32 on 4x64x16, level 0>) out(%7 : !cnm.buffer<64xf32 on 4x64x16, level 0>) on !cnm.workgroup<4x64x16> {
    ^bb0(%arg1: memref<64xf32>, %arg2: memref<f32>, %arg3: memref<64xf32>):
      %c0_13 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c1_14 = arith.constant 1 : index
      scf.for %arg4 = %c0_13 to %c64 step %c1_14 {
        %15 = memref.load %arg1[%arg4] : memref<64xf32>
        %16 = memref.load %arg2[] : memref<f32>
        %17 = arith.subf %15, %16 : f32
        memref.store %17, %arg3[%arg4] : memref<64xf32>
      }
    }
    %alloc_3 = memref.alloc() : memref<4096x64xf32>
    cnm.gather %7[#map6] of %2 into %alloc_3 : !cnm.buffer<64xf32 on 4x64x16, level 0> into memref<4096x64xf32>
    %8 = memref.get_global @__constant_1xi64_0 : memref<1xi64>
    %reshape_4 = memref.reshape %alloc_3(%8) : (memref<4096x64xf32>, memref<1xi64>) -> memref<262144xf32>
    cnm.free_workgroup %2 : !cnm.workgroup<4x64x16>
    %alloc_5 = memref.alloc() : memref<262144xf32>
    scf.for %arg1 = %c0 to %c262144 step %c1 {
      %15 = memref.load %reshape_4[%arg1] : memref<262144xf32>
      %16 = math.exp %15 : f32
      memref.store %16, %alloc_5[%arg1] : memref<262144xf32>
    }
    %9 = memref.get_global @__constant_xf32_0 : memref<f32>
    %alloc_6 = memref.alloc() : memref<256xf32>
    %alloc_7 = memref.alloc() : memref<f32>
    scf.for %arg1 = %c0 to %c256 step %c1 {
      %15 = arith.muli %arg1, %c1024 : index
      %subview = memref.subview %alloc_5[%15] [1024] [1] : memref<262144xf32> to memref<1024xf32, strided<[1], offset: ?>>
      memref.copy %9, %alloc_7 : memref<f32> to memref<f32>
      scf.for %arg2 = %c0 to %c1024 step %c1 {
        %17 = memref.load %subview[%arg2] : memref<1024xf32, strided<[1], offset: ?>>
        %18 = memref.load %alloc_7[] : memref<f32>
        %19 = arith.addf %17, %18 : f32
        memref.store %19, %alloc_7[] : memref<f32>
      }
      %16 = memref.load %alloc_7[] : memref<f32>
      memref.store %16, %alloc_6[%arg1] : memref<256xf32>
    }
    %alloc_8 = memref.alloc() : memref<f32>
    memref.copy %9, %alloc_8 : memref<f32> to memref<f32>
    scf.for %arg1 = %c0 to %c256 step %c1 {
      %15 = memref.load %alloc_6[%arg1] : memref<256xf32>
      %16 = memref.load %alloc_8[] : memref<f32>
      %17 = arith.addf %15, %16 : f32
      memref.store %17, %alloc_8[] : memref<f32>
    }
    %10 = memref.load %alloc_8[] : memref<f32>
    %11 = cnm.workgroup : !cnm.workgroup<4x64x16>
    %reshape_9 = memref.reshape %alloc_5(%3) : (memref<262144xf32>, memref<2xi64>) -> memref<4096x64xf32>
    %12 = cnm.alloc() for %11 : !cnm.buffer<64xf32 on 4x64x16, level 0>
    cnm.scatter %reshape_9 into %12[#map6] of %11 : memref<4096x64xf32> into !cnm.buffer<64xf32 on 4x64x16, level 0>
    %alloc_10 = memref.alloc() : memref<16xf32>
    memref.store %10, %alloc_10[%c0] : memref<16xf32>
    memref.store %10, %alloc_10[%c1] : memref<16xf32>
    memref.store %10, %alloc_10[%c2] : memref<16xf32>
    memref.store %10, %alloc_10[%c3] : memref<16xf32>
    memref.store %10, %alloc_10[%c4] : memref<16xf32>
    memref.store %10, %alloc_10[%c5] : memref<16xf32>
    memref.store %10, %alloc_10[%c6] : memref<16xf32>
    memref.store %10, %alloc_10[%c7] : memref<16xf32>
    memref.store %10, %alloc_10[%c8] : memref<16xf32>
    memref.store %10, %alloc_10[%c9] : memref<16xf32>
    memref.store %10, %alloc_10[%c10] : memref<16xf32>
    memref.store %10, %alloc_10[%c11] : memref<16xf32>
    memref.store %10, %alloc_10[%c12] : memref<16xf32>
    memref.store %10, %alloc_10[%c13] : memref<16xf32>
    memref.store %10, %alloc_10[%c14] : memref<16xf32>
    memref.store %10, %alloc_10[%c15] : memref<16xf32>
    %13 = cnm.alloc() for %11 : !cnm.buffer<f32 on 4x64x16, level 0>
    cnm.scatter %alloc_10 into %13[#map1] of %11 : memref<16xf32> into !cnm.buffer<f32 on 4x64x16, level 0>
    %14 = cnm.alloc() for %11 : !cnm.buffer<64xf32 on 4x64x16, level 0>
    cnm.scatter %6 into %14[#map6] of %11 : memref<4096x64xf32> into !cnm.buffer<64xf32 on 4x64x16, level 0>
    cnm.launch %11 in(%12, %13 : !cnm.buffer<64xf32 on 4x64x16, level 0>, !cnm.buffer<f32 on 4x64x16, level 0>) out(%14 : !cnm.buffer<64xf32 on 4x64x16, level 0>) on !cnm.workgroup<4x64x16> {
    ^bb0(%arg1: memref<64xf32>, %arg2: memref<f32>, %arg3: memref<64xf32>):
      %c0_13 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c1_14 = arith.constant 1 : index
      scf.for %arg4 = %c0_13 to %c64 step %c1_14 {
        %15 = memref.load %arg1[%arg4] : memref<64xf32>
        %16 = memref.load %arg2[] : memref<f32>
        %17 = arith.divf %15, %16 : f32
        memref.store %17, %arg3[%arg4] : memref<64xf32>
      }
    }
    %alloc_11 = memref.alloc() : memref<4096x64xf32>
    cnm.gather %14[#map6] of %11 into %alloc_11 : !cnm.buffer<64xf32 on 4x64x16, level 0> into memref<4096x64xf32>
    %reshape_12 = memref.reshape %alloc_11(%8) : (memref<4096x64xf32>, memref<1xi64>) -> memref<262144xf32>
    cnm.free_workgroup %11 : !cnm.workgroup<4x64x16>
    return %reshape_12 : memref<262144xf32>
  }

  func.func @rmsnorm_262144_opt(%arg0: memref<262144xf32>, %arg1: memref<262144xf32>) -> memref<262144xf32> {
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
    %c0f = arith.constant 0.0 : f32
    %cst = arith.constant 9.99999974E-6 : f32
    %cst_0 = arith.constant 2.621440e+05 : f32

    %reshape_pattern = memref.get_global @__constant_2xi64_0 : memref<2xi64>
    %v_reshape = memref.reshape %arg0(%reshape_pattern) : (memref<262144xf32>, memref<2xi64>) -> memref<4096x64xf32>

    %wg0 = cnm.workgroup : !cnm.workgroup<4x64x16>

    %2 = cnm.alloc() for %wg0 : !cnm.buffer<64xf32 on 4x64x16, level 0>
    cnm.scatter %v_reshape into %2[#map6] of %wg0 : memref<4096x64xf32> into !cnm.buffer<64xf32 on 4x64x16, level 0>

    %3 = cnm.alloc() for %wg0 : !cnm.buffer<64xf32 on 4x64x16, level 0>
    cnm.scatter %v_reshape into %3[#map6] of %wg0 : memref<4096x64xf32> into !cnm.buffer<64xf32 on 4x64x16, level 0>

    %4 = memref.get_global @__constant_4096x2xf32 : memref<4096x2xf32>
    %5 = cnm.alloc() for %wg0 : !cnm.buffer<2xf32 on 4x64x16, level 0>
    cnm.scatter %4 into %5[#map6] of %wg0 : memref<4096x2xf32> into !cnm.buffer<2xf32 on 4x64x16, level 0>

    cnm.launch %wg0 in(%2, %3 : !cnm.buffer<64xf32 on 4x64x16, level 0>, !cnm.buffer<64xf32 on 4x64x16, level 0>) out(%5 : !cnm.buffer<2xf32 on 4x64x16, level 0>) on !cnm.workgroup<4x64x16> {
    ^bb0(%a: memref<64xf32>, %b: memref<64xf32>, %c: memref<2xf32>):
      %c_0 = arith.constant 0 : index
      %c_1 = arith.constant 1 : index
      %c_2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      scf.for %i0 = %c0_12 to %c64 step %c2_13 {
        %i1 = arith.addi %i, %c_1 : index
        %a0 = memref.load %a[%i0] : memref<64xf32>
        %b0 = memref.load %b[%i0] : memref<64xf32>
        %c0 = memref.load %c[%c_0] : memref<2xf32>
        %p0 = arith.mulf %a0, %b0 : f32
        %r0 = arith.addf %p0, %c0 : f32
        memref.store %r0, %c[%c_0] : memref<64xf32>

        %a1 = memref.load %a[%i1] : memref<64xf32>
        %b1 = memref.load %b[%i1] : memref<64xf32>
        %c1 = memref.load %c[%c_1] : memref<2xf32>
        %p1 = arith.mulf %a1, %b1 : f32
        %r1 = arith.addf %p1, %c1 : f32
        memref.store %r1, %c[%c_1] : memref<64xf32>
      }
    }

    %alloc = memref.alloc() : memref<4096x2xf32>
    cnm.gather %5[#map6] of %wg0 into %alloc : !cnm.buffer<64xf32 on 4x64x16, level 0> into memref<4096x2xf32>

    cnm.free_workgroup %wg0 : !cnm.workgroup<4x64x16>

    %ss = scf.for %i = %c0 to %c4096 step %c1 iter_args(%acc = %c0f) {
      %a = memref.load %alloc[%i, %c0] : memref<4096x2xf32>
      %b = memref.load %alloc[%i, %c1] : memref<4096x2xf32>
      %c = arith.addf %a, %b : f32
      %r = arith.addf %c, %acc : f32
      scf.yield %r : f32
    }

    %9 = arith.divf %ss, %cst_0 : f32
    %10 = arith.addf %9, %cst : f32
    %11 = math.rsqrt %10 : f32

    %alloc_5 = memref.alloc() : memref<16xf32>
    memref.store %11, %alloc_5[%c0] : memref<16xf32>
    memref.store %11, %alloc_5[%c1] : memref<16xf32>
    memref.store %11, %alloc_5[%c2] : memref<16xf32>
    memref.store %11, %alloc_5[%c3] : memref<16xf32>
    memref.store %11, %alloc_5[%c4] : memref<16xf32>
    memref.store %11, %alloc_5[%c5] : memref<16xf32>
    memref.store %11, %alloc_5[%c6] : memref<16xf32>
    memref.store %11, %alloc_5[%c7] : memref<16xf32>
    memref.store %11, %alloc_5[%c8] : memref<16xf32>
    memref.store %11, %alloc_5[%c9] : memref<16xf32>
    memref.store %11, %alloc_5[%c10] : memref<16xf32>
    memref.store %11, %alloc_5[%c11] : memref<16xf32>
    memref.store %11, %alloc_5[%c12] : memref<16xf32>
    memref.store %11, %alloc_5[%c13] : memref<16xf32>
    memref.store %11, %alloc_5[%c14] : memref<16xf32>
    memref.store %11, %alloc_5[%c15] : memref<16xf32>

    %wg1 = cnm.workgroup : !cnm.workgroup<4x64x16>
    %13 = cnm.alloc() for %wg1 : !cnm.buffer<64xf32 on 4x64x16, level 0>
    cnm.scatter %v_reshape into %13[#map6] of %wg1 : memref<4096x64xf32> into !cnm.buffer<64xf32 on 4x64x16, level 0>

    %14 = cnm.alloc() for %wg1 : !cnm.buffer<f32 on 4x64x16, level 0>
    cnm.scatter %alloc_5 into %14[#map1] of %wg1 : memref<16xf32> into !cnm.buffer<f32 on 4x64x16, level 0>

    %15 = cnm.alloc() for %wg1 : !cnm.buffer<64xf32 on 4x64x16, level 0>
    cnm.scatter %4 into %15[#map6] of %wg1 : memref<4096x64xf32> into !cnm.buffer<64xf32 on 4x64x16, level 0>

    %reshape_9 = memref.reshape %arg1(%reshape_pattern) : (memref<262144xf32>, memref<2xi64>) -> memref<4096x64xf32>
    %18 = cnm.alloc() for %16 : !cnm.buffer<64xf32 on 4x64x16, level 0>
    cnm.scatter %reshape_9 into %18[#map6] of %16 : memref<4096x64xf32> into !cnm.buffer<64xf32 on 4x64x16, level 0>

    cnm.launch %wg1 in(%13, %14 : !cnm.buffer<64xf32 on 4x64x16, level 0>, !cnm.buffer<f32 on 4x64x16, level 0>) out(%15 : !cnm.buffer<64xf32 on 4x64x16, level 0>) on !cnm.workgroup<4x64x16> {
    ^bb0(%v_buf: memref<64xf32>, %w_buf: memref<64xf32>, %s_buf: memref<f32>, %r_buf: memref<64xf32>):
      %c_0 = arith.constant 0 : index
      %c_1 = arith.constant 1 : index
      %c_64 = arith.constant 64 : index
      %s = memref.load %s_buf[] : memref<f32>
      scf.for %i = %c_0 to %c_64 step %c_1 {
        %v = memref.load %arg2[%arg5] : memref<64xf32>
        %w = memref.load %arg2[%arg5] : memref<64xf32>
        %t = arith.mulf %v, %w : f32
        %r = arith.mulf %t, %s : f32
        memref.store %r, %r_buf[%i] : memref<64xf32>
      }
    }

    %alloc_6 = memref.alloc() : memref<4096x64xf32>
    cnm.gather %15[#map6] of %wg1 into %alloc_6 : !cnm.buffer<64xf32 on 4x64x16, level 0> into memref<4096x64xf32>
    %reshape_7 = memref.reshape %alloc_6(%6) : (memref<4096x64xf32>, memref<1xi64>) -> memref<262144xf32>

    cnm.free_workgroup %wg1 : !cnm.workgroup<4x64x16>

    return %reshape_7 : memref<262144xf32>
  }

  func.func @softmax_262144_opt(%arg0: memref<262144xf32>) -> memref<262144xf32> {
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
      %15 = arith.muli %arg1, %c1024 : index
      %subview = memref.subview %arg0[%15] [1024] [1] : memref<262144xf32> to memref<1024xf32, strided<[1], offset: ?>>
      memref.copy %0, %alloc_0 : memref<f32> to memref<f32>
      scf.for %arg2 = %c0 to %c1024 step %c1 {
        %17 = memref.load %subview[%arg2] : memref<1024xf32, strided<[1], offset: ?>>
        %18 = memref.load %alloc_0[] : memref<f32>
        %19 = arith.maximumf %17, %18 : f32
        memref.store %19, %alloc_0[] : memref<f32>
      }
      %16 = memref.load %alloc_0[] : memref<f32>
      memref.store %16, %alloc[%arg1] : memref<256xf32>
    }
    %alloc_1 = memref.alloc() : memref<f32>
    memref.copy %0, %alloc_1 : memref<f32> to memref<f32>
    scf.for %arg1 = %c0 to %c256 step %c1 {
      %15 = memref.load %alloc[%arg1] : memref<256xf32>
      %16 = memref.load %alloc_1[] : memref<f32>
      %17 = arith.maximumf %15, %16 : f32
      memref.store %17, %alloc_1[] : memref<f32>
    }
    %1 = memref.load %alloc_1[] : memref<f32>
    %2 = cnm.workgroup : !cnm.workgroup<4x64x16>
    %3 = memref.get_global @__constant_2xi64_0 : memref<2xi64>
    %reshape = memref.reshape %arg0(%3) : (memref<262144xf32>, memref<2xi64>) -> memref<4096x64xf32>
    %4 = cnm.alloc() for %2 : !cnm.buffer<64xf32 on 4x64x16, level 0>
    cnm.scatter %reshape into %4[#map6] of %2 : memref<4096x64xf32> into !cnm.buffer<64xf32 on 4x64x16, level 0>
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
    %5 = cnm.alloc() for %2 : !cnm.buffer<f32 on 4x64x16, level 0>
    cnm.scatter %alloc_2 into %5[#map1] of %2 : memref<16xf32> into !cnm.buffer<f32 on 4x64x16, level 0>
    %6 = memref.get_global @__constant_4096x64xf32 : memref<4096x64xf32>
    %7 = cnm.alloc() for %2 : !cnm.buffer<64xf32 on 4x64x16, level 0>
    cnm.scatter %6 into %7[#map6] of %2 : memref<4096x64xf32> into !cnm.buffer<64xf32 on 4x64x16, level 0>
    cnm.launch %2 in(%4, %5 : !cnm.buffer<64xf32 on 4x64x16, level 0>, !cnm.buffer<f32 on 4x64x16, level 0>) out(%7 : !cnm.buffer<64xf32 on 4x64x16, level 0>) on !cnm.workgroup<4x64x16> {
    ^bb0(%arg1: memref<64xf32>, %arg2: memref<f32>, %arg3: memref<64xf32>):
      %c0_13 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c1_14 = arith.constant 1 : index
      scf.for %arg4 = %c0_13 to %c64 step %c1_14 {
        %15 = memref.load %arg1[%arg4] : memref<64xf32>
        %16 = memref.load %arg2[] : memref<f32>
        %17 = arith.subf %15, %16 : f32
        memref.store %17, %arg3[%arg4] : memref<64xf32>
      }
    }
    %alloc_3 = memref.alloc() : memref<4096x64xf32>
    cnm.gather %7[#map6] of %2 into %alloc_3 : !cnm.buffer<64xf32 on 4x64x16, level 0> into memref<4096x64xf32>
    %8 = memref.get_global @__constant_1xi64_0 : memref<1xi64>
    %reshape_4 = memref.reshape %alloc_3(%8) : (memref<4096x64xf32>, memref<1xi64>) -> memref<262144xf32>
    cnm.free_workgroup %2 : !cnm.workgroup<4x64x16>
    %alloc_5 = memref.alloc() : memref<262144xf32>
    scf.for %arg1 = %c0 to %c262144 step %c1 {
      %15 = memref.load %reshape_4[%arg1] : memref<262144xf32>
      %16 = math.exp %15 : f32
      memref.store %16, %alloc_5[%arg1] : memref<262144xf32>
    }
    %9 = memref.get_global @__constant_xf32_0 : memref<f32>
    %alloc_6 = memref.alloc() : memref<256xf32>
    %alloc_7 = memref.alloc() : memref<f32>
    scf.for %arg1 = %c0 to %c256 step %c1 {
      %15 = arith.muli %arg1, %c1024 : index
      %subview = memref.subview %alloc_5[%15] [1024] [1] : memref<262144xf32> to memref<1024xf32, strided<[1], offset: ?>>
      memref.copy %9, %alloc_7 : memref<f32> to memref<f32>
      scf.for %arg2 = %c0 to %c1024 step %c1 {
        %17 = memref.load %subview[%arg2] : memref<1024xf32, strided<[1], offset: ?>>
        %18 = memref.load %alloc_7[] : memref<f32>
        %19 = arith.addf %17, %18 : f32
        memref.store %19, %alloc_7[] : memref<f32>
      }
      %16 = memref.load %alloc_7[] : memref<f32>
      memref.store %16, %alloc_6[%arg1] : memref<256xf32>
    }
    %alloc_8 = memref.alloc() : memref<f32>
    memref.copy %9, %alloc_8 : memref<f32> to memref<f32>
    scf.for %arg1 = %c0 to %c256 step %c1 {
      %15 = memref.load %alloc_6[%arg1] : memref<256xf32>
      %16 = memref.load %alloc_8[] : memref<f32>
      %17 = arith.addf %15, %16 : f32
      memref.store %17, %alloc_8[] : memref<f32>
    }
    %10 = memref.load %alloc_8[] : memref<f32>
    %11 = cnm.workgroup : !cnm.workgroup<4x64x16>
    %reshape_9 = memref.reshape %alloc_5(%3) : (memref<262144xf32>, memref<2xi64>) -> memref<4096x64xf32>
    %12 = cnm.alloc() for %11 : !cnm.buffer<64xf32 on 4x64x16, level 0>
    cnm.scatter %reshape_9 into %12[#map6] of %11 : memref<4096x64xf32> into !cnm.buffer<64xf32 on 4x64x16, level 0>
    %alloc_10 = memref.alloc() : memref<16xf32>
    memref.store %10, %alloc_10[%c0] : memref<16xf32>
    memref.store %10, %alloc_10[%c1] : memref<16xf32>
    memref.store %10, %alloc_10[%c2] : memref<16xf32>
    memref.store %10, %alloc_10[%c3] : memref<16xf32>
    memref.store %10, %alloc_10[%c4] : memref<16xf32>
    memref.store %10, %alloc_10[%c5] : memref<16xf32>
    memref.store %10, %alloc_10[%c6] : memref<16xf32>
    memref.store %10, %alloc_10[%c7] : memref<16xf32>
    memref.store %10, %alloc_10[%c8] : memref<16xf32>
    memref.store %10, %alloc_10[%c9] : memref<16xf32>
    memref.store %10, %alloc_10[%c10] : memref<16xf32>
    memref.store %10, %alloc_10[%c11] : memref<16xf32>
    memref.store %10, %alloc_10[%c12] : memref<16xf32>
    memref.store %10, %alloc_10[%c13] : memref<16xf32>
    memref.store %10, %alloc_10[%c14] : memref<16xf32>
    memref.store %10, %alloc_10[%c15] : memref<16xf32>
    %13 = cnm.alloc() for %11 : !cnm.buffer<f32 on 4x64x16, level 0>
    cnm.scatter %alloc_10 into %13[#map1] of %11 : memref<16xf32> into !cnm.buffer<f32 on 4x64x16, level 0>
    %14 = cnm.alloc() for %11 : !cnm.buffer<64xf32 on 4x64x16, level 0>
    cnm.scatter %6 into %14[#map6] of %11 : memref<4096x64xf32> into !cnm.buffer<64xf32 on 4x64x16, level 0>
    cnm.launch %11 in(%12, %13 : !cnm.buffer<64xf32 on 4x64x16, level 0>, !cnm.buffer<f32 on 4x64x16, level 0>) out(%14 : !cnm.buffer<64xf32 on 4x64x16, level 0>) on !cnm.workgroup<4x64x16> {
    ^bb0(%arg1: memref<64xf32>, %arg2: memref<f32>, %arg3: memref<64xf32>):
      %c0_13 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c1_14 = arith.constant 1 : index
      scf.for %arg4 = %c0_13 to %c64 step %c1_14 {
        %15 = memref.load %arg1[%arg4] : memref<64xf32>
        %16 = memref.load %arg2[] : memref<f32>
        %17 = arith.divf %15, %16 : f32
        memref.store %17, %arg3[%arg4] : memref<64xf32>
      }
    }
    %alloc_11 = memref.alloc() : memref<4096x64xf32>
    cnm.gather %14[#map6] of %11 into %alloc_11 : !cnm.buffer<64xf32 on 4x64x16, level 0> into memref<4096x64xf32>
    %reshape_12 = memref.reshape %alloc_11(%8) : (memref<4096x64xf32>, memref<1xi64>) -> memref<262144xf32>
    cnm.free_workgroup %11 : !cnm.workgroup<4x64x16>
    return %reshape_12 : memref<262144xf32>
  }
  func.func @va_262144(%arg0: memref<262144xf32>, %arg1: memref<262144xf32>) -> memref<262144xf32> {
    %0 = cnm.workgroup : !cnm.workgroup<4x64x16>
    %1 = memref.get_global @__constant_2xi64_0 : memref<2xi64>
    %reshape = memref.reshape %arg0(%1) : (memref<262144xf32>, memref<2xi64>) -> memref<4096x64xf32>
    %2 = cnm.alloc() for %0 : !cnm.buffer<64xf32 on 4x64x16, level 0>
    cnm.scatter %reshape into %2[#map6] of %0 : memref<4096x64xf32> into !cnm.buffer<64xf32 on 4x64x16, level 0>
    %reshape_0 = memref.reshape %arg1(%1) : (memref<262144xf32>, memref<2xi64>) -> memref<4096x64xf32>
    %3 = cnm.alloc() for %0 : !cnm.buffer<64xf32 on 4x64x16, level 0>
    cnm.scatter %reshape_0 into %3[#map6] of %0 : memref<4096x64xf32> into !cnm.buffer<64xf32 on 4x64x16, level 0>
    %4 = memref.get_global @__constant_4096x64xf32 : memref<4096x64xf32>
    %5 = cnm.alloc() for %0 : !cnm.buffer<64xf32 on 4x64x16, level 0>
    cnm.scatter %4 into %5[#map6] of %0 : memref<4096x64xf32> into !cnm.buffer<64xf32 on 4x64x16, level 0>
    cnm.launch %0 in(%2, %3 : !cnm.buffer<64xf32 on 4x64x16, level 0>, !cnm.buffer<64xf32 on 4x64x16, level 0>) out(%5 : !cnm.buffer<64xf32 on 4x64x16, level 0>) on !cnm.workgroup<4x64x16> {
    ^bb0(%arg2: memref<64xf32>, %arg3: memref<64xf32>, %arg4: memref<64xf32>):
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c1 = arith.constant 1 : index
      scf.for %arg5 = %c0 to %c64 step %c1 {
        %7 = memref.load %arg2[%arg5] : memref<64xf32>
        %8 = memref.load %arg3[%arg5] : memref<64xf32>
        %9 = arith.addf %7, %8 : f32
        memref.store %9, %arg4[%arg5] : memref<64xf32>
      }
    }
    %alloc = memref.alloc() : memref<4096x64xf32>
    cnm.gather %5[#map6] of %0 into %alloc : !cnm.buffer<64xf32 on 4x64x16, level 0> into memref<4096x64xf32>
    %6 = memref.get_global @__constant_1xi64_0 : memref<1xi64>
    %reshape_1 = memref.reshape %alloc(%6) : (memref<4096x64xf32>, memref<1xi64>) -> memref<262144xf32>
    cnm.free_workgroup %0 : !cnm.workgroup<4x64x16>
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
        %28 = cnm.workgroup : !cnm.workgroup<1x16x16>
        %29 = memref.get_global @__constant_2xi64_4 : memref<2xi64>
        memref.copy %subview_31, %alloc_5 : memref<4096xf32, strided<[1], offset: ?>> to memref<4096xf32>
        %reshape_33 = memref.reshape %alloc_5(%29) : (memref<4096xf32>, memref<2xi64>) -> memref<256x16xf32>
        %30 = cnm.alloc() for %28 : !cnm.buffer<16xf32 on 1x16x16, level 0>
        cnm.scatter %reshape_33 into %30[#map3] of %28 : memref<256x16xf32> into !cnm.buffer<16xf32 on 1x16x16, level 0>
        memref.copy %subview_32, %alloc_6 : memref<4096xf32, strided<[1], offset: ?>> to memref<4096xf32>
        %reshape_34 = memref.reshape %alloc_6(%29) : (memref<4096xf32>, memref<2xi64>) -> memref<256x16xf32>
        %31 = cnm.alloc() for %28 : !cnm.buffer<16xf32 on 1x16x16, level 0>
        cnm.scatter %reshape_34 into %31[#map3] of %28 : memref<256x16xf32> into !cnm.buffer<16xf32 on 1x16x16, level 0>
        %32 = memref.get_global @__constant_256x16xf32 : memref<256x16xf32>
        %33 = cnm.alloc() for %28 : !cnm.buffer<16xf32 on 1x16x16, level 0>
        cnm.scatter %32 into %33[#map3] of %28 : memref<256x16xf32> into !cnm.buffer<16xf32 on 1x16x16, level 0>
        cnm.launch %28 in(%30, %31 : !cnm.buffer<16xf32 on 1x16x16, level 0>, !cnm.buffer<16xf32 on 1x16x16, level 0>) out(%33 : !cnm.buffer<16xf32 on 1x16x16, level 0>) on !cnm.workgroup<1x16x16> {
        ^bb0(%arg8: memref<16xf32>, %arg9: memref<16xf32>, %arg10: memref<16xf32>):
          %c0_37 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_38 = arith.constant 1 : index
          scf.for %arg11 = %c0_37 to %c16 step %c1_38 {
            %38 = memref.load %arg8[%arg11] : memref<16xf32>
            %39 = memref.load %arg9[%arg11] : memref<16xf32>
            %40 = arith.mulf %38, %39 : f32
            memref.store %40, %arg10[%arg11] : memref<16xf32>
          }
        }
        cnm.gather %33[#map3] of %28 into %alloc_7 : !cnm.buffer<16xf32 on 1x16x16, level 0> into memref<256x16xf32>
        %34 = memref.get_global @__constant_1xi64_4 : memref<1xi64>
        %reshape_35 = memref.reshape %alloc_7(%34) : (memref<256x16xf32>, memref<1xi64>) -> memref<4096xf32>
        cnm.free_workgroup %28 : !cnm.workgroup<1x16x16>
        %35 = memref.get_global @__constant_xf32_0 : memref<f32>
        scf.for %arg8 = %c0 to %c4 step %c1 {
          %38 = arith.muli %arg8, %c1024 : index
          %subview_37 = memref.subview %reshape_35[%38] [1024] [1] : memref<4096xf32> to memref<1024xf32, strided<[1], offset: ?>>
          memref.copy %35, %alloc_9 : memref<f32> to memref<f32>
          scf.for %arg9 = %c0 to %c1024 step %c1 {
            %40 = memref.load %subview_37[%arg9] : memref<1024xf32, strided<[1], offset: ?>>
            %41 = memref.load %alloc_9[] : memref<f32>
            %42 = arith.addf %40, %41 : f32
            memref.store %42, %alloc_9[] : memref<f32>
          }
          %39 = memref.load %alloc_9[] : memref<f32>
          memref.store %39, %alloc_8[%arg8] : memref<4xf32>
        }
        memref.copy %35, %alloc_10 : memref<f32> to memref<f32>
        scf.for %arg8 = %c0 to %c4 step %c1 {
          %38 = memref.load %alloc_8[%arg8] : memref<4xf32>
          %39 = memref.load %alloc_10[] : memref<f32>
          %40 = arith.addf %38, %39 : f32
          memref.store %40, %alloc_10[] : memref<f32>
        }
        %36 = memref.load %alloc_10[] : memref<f32>
        %37 = arith.divf %36, %cst_0 : f32
        %alloc_36 = memref.alloc() : memref<1024xf32>
        memref.copy %arg7, %alloc_36 : memref<1024xf32> to memref<1024xf32>
        memref.store %37, %alloc_36[%arg6] : memref<1024xf32>
        scf.yield %alloc_36 : memref<1024xf32>
      }
      %4 = memref.get_global @__constant_xf32 : memref<f32>
      memref.copy %4, %alloc_12 : memref<f32> to memref<f32>
      scf.for %arg6 = %c0 to %c1024 step %c1 {
        %28 = memref.load %3[%arg6] : memref<1024xf32>
        %29 = memref.load %alloc_12[] : memref<f32>
        %30 = arith.maximumf %28, %29 : f32
        memref.store %30, %alloc_12[] : memref<f32>
      }
      %5 = memref.load %alloc_12[] : memref<f32>
      memref.store %5, %alloc_11[%c0] : memref<1xf32>
      memref.copy %4, %alloc_13 : memref<f32> to memref<f32>
      %6 = memref.load %alloc_11[%c0] : memref<1xf32>
      %7 = memref.load %alloc_13[] : memref<f32>
      %8 = arith.maximumf %6, %7 : f32
      memref.store %8, %alloc_13[] : memref<f32>
      %9 = memref.load %alloc_13[] : memref<f32>
      %10 = cnm.workgroup : !cnm.workgroup<1x16x16>
      %11 = memref.get_global @__constant_2xi64_5 : memref<2xi64>
      %reshape = memref.reshape %3(%11) : (memref<1024xf32>, memref<2xi64>) -> memref<256x4xf32>
      %12 = cnm.alloc() for %10 : !cnm.buffer<4xf32 on 1x16x16, level 0>
      cnm.scatter %reshape into %12[#map3] of %10 : memref<256x4xf32> into !cnm.buffer<4xf32 on 1x16x16, level 0>
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
      %13 = cnm.alloc() for %10 : !cnm.buffer<f32 on 1x16x16, level 0>
      cnm.scatter %alloc_14 into %13[#map1] of %10 : memref<16xf32> into !cnm.buffer<f32 on 1x16x16, level 0>
      %14 = memref.get_global @__constant_256x4xf32 : memref<256x4xf32>
      %15 = cnm.alloc() for %10 : !cnm.buffer<4xf32 on 1x16x16, level 0>
      cnm.scatter %14 into %15[#map3] of %10 : memref<256x4xf32> into !cnm.buffer<4xf32 on 1x16x16, level 0>
      cnm.launch %10 in(%12, %13 : !cnm.buffer<4xf32 on 1x16x16, level 0>, !cnm.buffer<f32 on 1x16x16, level 0>) out(%15 : !cnm.buffer<4xf32 on 1x16x16, level 0>) on !cnm.workgroup<1x16x16> {
      ^bb0(%arg6: memref<4xf32>, %arg7: memref<f32>, %arg8: memref<4xf32>):
        %c0_31 = arith.constant 0 : index
        %c4_32 = arith.constant 4 : index
        %c1_33 = arith.constant 1 : index
        scf.for %arg9 = %c0_31 to %c4_32 step %c1_33 {
          %28 = memref.load %arg6[%arg9] : memref<4xf32>
          %29 = memref.load %arg7[] : memref<f32>
          %30 = arith.subf %28, %29 : f32
          memref.store %30, %arg8[%arg9] : memref<4xf32>
        }
      }
      cnm.gather %15[#map3] of %10 into %alloc_15 : !cnm.buffer<4xf32 on 1x16x16, level 0> into memref<256x4xf32>
      %16 = memref.get_global @__constant_1xi64_5 : memref<1xi64>
      %reshape_27 = memref.reshape %alloc_15(%16) : (memref<256x4xf32>, memref<1xi64>) -> memref<1024xf32>
      cnm.free_workgroup %10 : !cnm.workgroup<1x16x16>
      scf.for %arg6 = %c0 to %c1024 step %c1 {
        %28 = memref.load %reshape_27[%arg6] : memref<1024xf32>
        %29 = math.exp %28 : f32
        memref.store %29, %alloc_16[%arg6] : memref<1024xf32>
      }
      %17 = memref.get_global @__constant_xf32_0 : memref<f32>
      memref.copy %17, %alloc_18 : memref<f32> to memref<f32>
      scf.for %arg6 = %c0 to %c1024 step %c1 {
        %28 = memref.load %alloc_16[%arg6] : memref<1024xf32>
        %29 = memref.load %alloc_18[] : memref<f32>
        %30 = arith.addf %28, %29 : f32
        memref.store %30, %alloc_18[] : memref<f32>
      }
      %18 = memref.load %alloc_18[] : memref<f32>
      memref.store %18, %alloc_17[%c0] : memref<1xf32>
      memref.copy %17, %alloc_19 : memref<f32> to memref<f32>
      %19 = memref.load %alloc_17[%c0] : memref<1xf32>
      %20 = memref.load %alloc_19[] : memref<f32>
      %21 = arith.addf %19, %20 : f32
      memref.store %21, %alloc_19[] : memref<f32>
      %22 = memref.load %alloc_19[] : memref<f32>
      %23 = cnm.workgroup : !cnm.workgroup<1x16x16>
      %reshape_28 = memref.reshape %alloc_16(%11) : (memref<1024xf32>, memref<2xi64>) -> memref<256x4xf32>
      %24 = cnm.alloc() for %23 : !cnm.buffer<4xf32 on 1x16x16, level 0>
      cnm.scatter %reshape_28 into %24[#map3] of %23 : memref<256x4xf32> into !cnm.buffer<4xf32 on 1x16x16, level 0>
      memref.store %22, %alloc_20[%c0] : memref<16xf32>
      memref.store %22, %alloc_20[%c1] : memref<16xf32>
      memref.store %22, %alloc_20[%c2] : memref<16xf32>
      memref.store %22, %alloc_20[%c3] : memref<16xf32>
      memref.store %22, %alloc_20[%c4] : memref<16xf32>
      memref.store %22, %alloc_20[%c5] : memref<16xf32>
      memref.store %22, %alloc_20[%c6] : memref<16xf32>
      memref.store %22, %alloc_20[%c7] : memref<16xf32>
      memref.store %22, %alloc_20[%c8] : memref<16xf32>
      memref.store %22, %alloc_20[%c9] : memref<16xf32>
      memref.store %22, %alloc_20[%c10] : memref<16xf32>
      memref.store %22, %alloc_20[%c11] : memref<16xf32>
      memref.store %22, %alloc_20[%c12] : memref<16xf32>
      memref.store %22, %alloc_20[%c13] : memref<16xf32>
      memref.store %22, %alloc_20[%c14] : memref<16xf32>
      memref.store %22, %alloc_20[%c15] : memref<16xf32>
      %25 = cnm.alloc() for %23 : !cnm.buffer<f32 on 1x16x16, level 0>
      cnm.scatter %alloc_20 into %25[#map1] of %23 : memref<16xf32> into !cnm.buffer<f32 on 1x16x16, level 0>
      %26 = cnm.alloc() for %23 : !cnm.buffer<4xf32 on 1x16x16, level 0>
      cnm.scatter %14 into %26[#map3] of %23 : memref<256x4xf32> into !cnm.buffer<4xf32 on 1x16x16, level 0>
      cnm.launch %23 in(%24, %25 : !cnm.buffer<4xf32 on 1x16x16, level 0>, !cnm.buffer<f32 on 1x16x16, level 0>) out(%26 : !cnm.buffer<4xf32 on 1x16x16, level 0>) on !cnm.workgroup<1x16x16> {
      ^bb0(%arg6: memref<4xf32>, %arg7: memref<f32>, %arg8: memref<4xf32>):
        %c0_31 = arith.constant 0 : index
        %c4_32 = arith.constant 4 : index
        %c1_33 = arith.constant 1 : index
        scf.for %arg9 = %c0_31 to %c4_32 step %c1_33 {
          %28 = memref.load %arg6[%arg9] : memref<4xf32>
          %29 = memref.load %arg7[] : memref<f32>
          %30 = arith.divf %28, %29 : f32
          memref.store %30, %arg8[%arg9] : memref<4xf32>
        }
      }
      cnm.gather %26[#map3] of %23 into %alloc_21 : !cnm.buffer<4xf32 on 1x16x16, level 0> into memref<256x4xf32>
      %reshape_29 = memref.reshape %alloc_21(%16) : (memref<256x4xf32>, memref<1xi64>) -> memref<1024xf32>
      cnm.free_workgroup %23 : !cnm.workgroup<1x16x16>
      scf.for %arg6 = %c0 to %c4096 step %c1 {
        memref.store %cst, %alloc_22[%arg6] : memref<4096xf32>
      }
      memref.copy %alloc_22, %alloc_23 : memref<4096xf32> to memref<4096xf32>
      %27 = scf.for %arg6 = %c0 to %0 step %c1 iter_args(%arg7 = %alloc_23) -> (memref<4096xf32>) {
        %subview_31 = memref.subview %arg2[%arg6, %2] [1, 4096] [1, 1] : memref<1024x32768xf32> to memref<4096xf32, strided<[1], offset: ?>>
        %28 = memref.load %reshape_29[%arg6] : memref<1024xf32>
        %29 = cnm.workgroup : !cnm.workgroup<1x16x16>
        %30 = memref.get_global @__constant_2xi64_4 : memref<2xi64>
        memref.copy %subview_31, %alloc_24 : memref<4096xf32, strided<[1], offset: ?>> to memref<4096xf32>
        %reshape_32 = memref.reshape %alloc_24(%30) : (memref<4096xf32>, memref<2xi64>) -> memref<256x16xf32>
        %31 = cnm.alloc() for %29 : !cnm.buffer<16xf32 on 1x16x16, level 0>
        cnm.scatter %reshape_32 into %31[#map3] of %29 : memref<256x16xf32> into !cnm.buffer<16xf32 on 1x16x16, level 0>
        memref.store %28, %alloc_25[%c0] : memref<16xf32>
        memref.store %28, %alloc_25[%c1] : memref<16xf32>
        memref.store %28, %alloc_25[%c2] : memref<16xf32>
        memref.store %28, %alloc_25[%c3] : memref<16xf32>
        memref.store %28, %alloc_25[%c4] : memref<16xf32>
        memref.store %28, %alloc_25[%c5] : memref<16xf32>
        memref.store %28, %alloc_25[%c6] : memref<16xf32>
        memref.store %28, %alloc_25[%c7] : memref<16xf32>
        memref.store %28, %alloc_25[%c8] : memref<16xf32>
        memref.store %28, %alloc_25[%c9] : memref<16xf32>
        memref.store %28, %alloc_25[%c10] : memref<16xf32>
        memref.store %28, %alloc_25[%c11] : memref<16xf32>
        memref.store %28, %alloc_25[%c12] : memref<16xf32>
        memref.store %28, %alloc_25[%c13] : memref<16xf32>
        memref.store %28, %alloc_25[%c14] : memref<16xf32>
        memref.store %28, %alloc_25[%c15] : memref<16xf32>
        %32 = cnm.alloc() for %29 : !cnm.buffer<f32 on 1x16x16, level 0>
        cnm.scatter %alloc_25 into %32[#map1] of %29 : memref<16xf32> into !cnm.buffer<f32 on 1x16x16, level 0>
        %33 = memref.get_global @__constant_256x16xf32 : memref<256x16xf32>
        %34 = cnm.alloc() for %29 : !cnm.buffer<16xf32 on 1x16x16, level 0>
        cnm.scatter %33 into %34[#map3] of %29 : memref<256x16xf32> into !cnm.buffer<16xf32 on 1x16x16, level 0>
        cnm.launch %29 in(%31, %32 : !cnm.buffer<16xf32 on 1x16x16, level 0>, !cnm.buffer<f32 on 1x16x16, level 0>) out(%34 : !cnm.buffer<16xf32 on 1x16x16, level 0>) on !cnm.workgroup<1x16x16> {
        ^bb0(%arg8: memref<16xf32>, %arg9: memref<f32>, %arg10: memref<16xf32>):
          %c0_38 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_39 = arith.constant 1 : index
          scf.for %arg11 = %c0_38 to %c16 step %c1_39 {
            %40 = memref.load %arg8[%arg11] : memref<16xf32>
            %41 = memref.load %arg9[] : memref<f32>
            %42 = arith.mulf %40, %41 : f32
            memref.store %42, %arg10[%arg11] : memref<16xf32>
          }
        }
        cnm.gather %34[#map3] of %29 into %alloc_26 : !cnm.buffer<16xf32 on 1x16x16, level 0> into memref<256x16xf32>
        %35 = memref.get_global @__constant_1xi64_4 : memref<1xi64>
        %reshape_33 = memref.reshape %alloc_26(%35) : (memref<256x16xf32>, memref<1xi64>) -> memref<4096xf32>
        cnm.free_workgroup %29 : !cnm.workgroup<1x16x16>
        %36 = cnm.workgroup : !cnm.workgroup<1x16x16>
        %reshape_34 = memref.reshape %arg7(%30) : (memref<4096xf32>, memref<2xi64>) -> memref<256x16xf32>
        %37 = cnm.alloc() for %36 : !cnm.buffer<16xf32 on 1x16x16, level 0>
        cnm.scatter %reshape_34 into %37[#map3] of %36 : memref<256x16xf32> into !cnm.buffer<16xf32 on 1x16x16, level 0>
        %reshape_35 = memref.reshape %reshape_33(%30) : (memref<4096xf32>, memref<2xi64>) -> memref<256x16xf32>
        %38 = cnm.alloc() for %36 : !cnm.buffer<16xf32 on 1x16x16, level 0>
        cnm.scatter %reshape_35 into %38[#map3] of %36 : memref<256x16xf32> into !cnm.buffer<16xf32 on 1x16x16, level 0>
        %39 = cnm.alloc() for %36 : !cnm.buffer<16xf32 on 1x16x16, level 0>
        cnm.scatter %33 into %39[#map3] of %36 : memref<256x16xf32> into !cnm.buffer<16xf32 on 1x16x16, level 0>
        cnm.launch %36 in(%37, %38 : !cnm.buffer<16xf32 on 1x16x16, level 0>, !cnm.buffer<16xf32 on 1x16x16, level 0>) out(%39 : !cnm.buffer<16xf32 on 1x16x16, level 0>) on !cnm.workgroup<1x16x16> {
        ^bb0(%arg8: memref<16xf32>, %arg9: memref<16xf32>, %arg10: memref<16xf32>):
          %c0_38 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_39 = arith.constant 1 : index
          scf.for %arg11 = %c0_38 to %c16 step %c1_39 {
            %40 = memref.load %arg8[%arg11] : memref<16xf32>
            %41 = memref.load %arg9[%arg11] : memref<16xf32>
            %42 = arith.addf %40, %41 : f32
            memref.store %42, %arg10[%arg11] : memref<16xf32>
          }
        }
        %alloc_36 = memref.alloc() : memref<256x16xf32>
        cnm.gather %39[#map3] of %36 into %alloc_36 : !cnm.buffer<16xf32 on 1x16x16, level 0> into memref<256x16xf32>
        %reshape_37 = memref.reshape %alloc_36(%35) : (memref<256x16xf32>, memref<1xi64>) -> memref<4096xf32>
        cnm.free_workgroup %36 : !cnm.workgroup<1x16x16>
        scf.yield %reshape_37 : memref<4096xf32>
      }
      %alloc_30 = memref.alloc() : memref<32768xf32>
      memref.copy %arg5, %alloc_30 : memref<32768xf32> to memref<32768xf32>
      %subview = memref.subview %alloc_30[%2] [4096] [1] : memref<32768xf32> to memref<4096xf32, strided<[1], offset: ?>>
      memref.copy %27, %subview : memref<4096xf32> to memref<4096xf32, strided<[1], offset: ?>>
      scf.yield %alloc_30 : memref<32768xf32>
    }
    return %1 : memref<32768xf32>
  }
}
