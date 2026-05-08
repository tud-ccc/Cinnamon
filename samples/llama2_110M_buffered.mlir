#map = affine_map<() -> ()>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> ()>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d1)>
#map5 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @forward(%arg0: index, %arg1: index, %arg2: memref<6x1024x768xf32>, %arg3: memref<6x1024x768xf32>, %arg4: memref<32000x768xf32>, %arg5: memref<6x768xf32>, %arg6: memref<6x768x768xf32>, %arg7: memref<6x768x768xf32>, %arg8: memref<6x768x768xf32>, %arg9: memref<6x768x768xf32>, %arg10: memref<6x2048x768xf32>, %arg11: memref<6x768x2048xf32>, %arg12: memref<6x2048x768xf32>, %arg13: memref<6x768xf32>, %arg14: memref<768xf32>, %arg15: memref<32000x768xf32>, %arg16: memref<32000xf32, strided<[1]>>) {
    %cst = arith.constant 1.000000e+04 : f32
    %cst_0 = arith.constant 4.800000e+01 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %c768 = arith.constant 768 : index
    %c48 = arith.constant 48 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %cst_2 = arith.constant 9.99999974E-6 : f32
    %cst_3 = arith.constant 7.680000e+02 : f32
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    %c1024 = arith.constant 1024 : index
    %cst_4 = arith.constant 6.92820311 : f32
    %cst_5 = arith.constant 0xFF800000 : f32
    %cst_6 = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<34048xf32>
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<2048xf32>
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<2048xf32>
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<2048xf32>
    %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<2048xf32>
    %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<2048xf32>
    %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<2048xf32>
    %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    %alloc_18 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    %alloc_19 = memref.alloc() {alignment = 64 : i64} : memref<2048xf32>
    %alloc_20 = memref.alloc() {alignment = 64 : i64} : memref<2048xf32>
    %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    %alloc_22 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<2048xf32>
    %alloc_24 = memref.alloc() {alignment = 64 : i64} : memref<2048xf32>
    %alloc_25 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    %alloc_26 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    %alloc_27 = memref.alloc() {alignment = 64 : i64} : memref<2048xf32>
    %alloc_28 = memref.alloc() {alignment = 64 : i64} : memref<2048xf32>
    %alloc_29 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    %alloc_30 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    %alloc_31 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    %subview = memref.subview %arg4[%arg0, 0] [1, 768] [1, 1] : memref<32000x768xf32> to memref<768xf32, strided<[1], offset: ?>>
    %subview_32 = memref.subview %arg5[0, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1]>>
    cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %alloc_116 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_116 : memref<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_6 : f32
      }
      linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%subview : memref<768xf32, strided<[1], offset: ?>>) outs(%alloc_116 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %8 = arith.mulf %in, %in : f32
        %9 = arith.addf %8, %out : f32
        linalg.yield %9 : f32
      }
      %4 = memref.load %alloc_116[] : memref<f32>
      %5 = arith.divf %4, %cst_3 : f32
      %6 = arith.addf %5, %cst_2 : f32
      %7 = math.rsqrt %6 : f32
      linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%subview, %subview_32 : memref<768xf32, strided<[1], offset: ?>>, memref<768xf32, strided<[1]>>) outs(%alloc_31 : memref<768xf32>) {
      ^bb0(%in: f32, %in_117: f32, %out: f32):
        %8 = arith.mulf %in, %7 : f32
        %9 = arith.mulf %8, %in_117 : f32
        linalg.yield %9 : f32
      }
      cinm.yield
    }
    %subview_33 = memref.subview %arg6[0, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1]>>
    %subview_34 = memref.subview %arg7[0, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1]>>
    %subview_35 = memref.subview %arg8[0, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1]>>
    cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %alloc_116 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
      linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%alloc_116 : memref<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_6 : f32
      }
      memref.copy %alloc_116, %alloc_30 : memref<768xf32> to memref<768xf32>
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_33, %alloc_31 : memref<768x768xf32, strided<[768, 1]>>, memref<768xf32>) outs(%alloc_30 : memref<768xf32>) {
      ^bb0(%in: f32, %in_120: f32, %out: f32):
        %4 = arith.mulf %in, %in_120 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
      %alloc_117 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
      memref.copy %alloc_116, %alloc_117 : memref<768xf32> to memref<768xf32>
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_34, %alloc_31 : memref<768x768xf32, strided<[768, 1]>>, memref<768xf32>) outs(%alloc_117 : memref<768xf32>) {
      ^bb0(%in: f32, %in_120: f32, %out: f32):
        %4 = arith.mulf %in, %in_120 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_35, %alloc_31 : memref<768x768xf32, strided<[768, 1]>>, memref<768xf32>) outs(%alloc_116 : memref<768xf32>) {
      ^bb0(%in: f32, %in_120: f32, %out: f32):
        %4 = arith.mulf %in, %in_120 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
      %subview_118 = memref.subview %arg2[0, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
      memref.copy %alloc_117, %subview_118 : memref<768xf32> to memref<768xf32, strided<[1], offset: ?>>
      %subview_119 = memref.subview %arg3[0, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
      memref.copy %alloc_116, %subview_119 : memref<768xf32> to memref<768xf32, strided<[1], offset: ?>>
      cinm.yield
    }
    %0 = arith.index_cast %arg1 : index to i64
    %1 = arith.uitofp %0 : i64 to f32
    %subview_36 = memref.subview %arg2[0, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
    scf.for %arg17 = %c0 to %c768 step %c2 {
      %4 = arith.remui %arg17, %c48 : index
      %5 = arith.index_cast %4 : index to i64
      %6 = arith.uitofp %5 : i64 to f32
      %7 = arith.divf %6, %cst_0 : f32
      %8 = math.powf %cst, %7 : f32
      %9 = arith.divf %cst_1, %8 : f32
      %10 = arith.mulf %1, %9 : f32
      %11 = math.cos %10 : f32
      %12 = math.sin %10 : f32
      %13 = arith.addi %arg17, %c1 : index
      %14 = memref.load %alloc_30[%arg17] : memref<768xf32>
      %15 = memref.load %alloc_30[%13] : memref<768xf32>
      %16 = arith.mulf %14, %11 : f32
      %17 = arith.mulf %15, %12 : f32
      %18 = arith.subf %16, %17 : f32
      memref.store %18, %alloc_30[%arg17] : memref<768xf32>
      %19 = arith.mulf %14, %12 : f32
      %20 = arith.mulf %15, %11 : f32
      %21 = arith.addf %19, %20 : f32
      memref.store %21, %alloc_30[%13] : memref<768xf32>
      %22 = arith.cmpi ult, %arg17, %c768 : index
      scf.if %22 {
        %23 = memref.load %subview_36[%arg17] : memref<768xf32, strided<[1], offset: ?>>
        %24 = memref.load %subview_36[%13] : memref<768xf32, strided<[1], offset: ?>>
        %25 = arith.mulf %23, %11 : f32
        %26 = arith.mulf %24, %12 : f32
        %27 = arith.subf %25, %26 : f32
        memref.store %27, %subview_36[%arg17] : memref<768xf32, strided<[1], offset: ?>>
        %28 = arith.mulf %23, %12 : f32
        %29 = arith.mulf %24, %11 : f32
        %30 = arith.addf %28, %29 : f32
        memref.store %30, %subview_36[%13] : memref<768xf32, strided<[1], offset: ?>>
      }
    }
    %subview_37 = memref.subview %arg2[0, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1]>>
    %subview_38 = memref.subview %arg3[0, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1]>>
    %2 = arith.addi %arg1, %c1 : index
    %alloc_39 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    %alloc_40 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    %alloc_41 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc_30, %alloc_41 : memref<768xf32> to memref<768xf32>
    scf.for %arg17 = %c0 to %c6 step %c1 {
      %4 = arith.muli %arg17, %c48 : index
      %alloc_116 = memref.alloc() {alignment = 64 : i64} : memref<1024xf32>
      scf.for %arg18 = %c0 to %2 step %c1 {
        %subview_118 = memref.subview %alloc_30[%4] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: ?>>
        %subview_119 = memref.subview %subview_37[%arg18, %4] [1, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<48xf32, strided<[1], offset: ?>>
        %5 = cinm.compute -> f32 attributes {workgroupShape = array<i64: 1, 1, 8>} {
          %alloc_120 = memref.alloc() {alignment = 64 : i64} : memref<f32>
          linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_120 : memref<f32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_6 : f32
          }
          linalg.generic {indexing_maps = [#map1, #map1, #map2], iterator_types = ["reduction"]} ins(%subview_118, %subview_119 : memref<48xf32, strided<[1], offset: ?>>, memref<48xf32, strided<[1], offset: ?>>) outs(%alloc_120 : memref<f32>) {
          ^bb0(%in: f32, %in_121: f32, %out: f32):
            %8 = arith.mulf %in, %in_121 : f32
            %9 = arith.addf %8, %out : f32
            linalg.yield %9 : f32
          }
          %6 = memref.load %alloc_120[] : memref<f32>
          %7 = arith.divf %6, %cst_4 : f32
          cinm.yield %7 : f32
        }
        memref.store %5, %alloc_116[%arg18] : memref<1024xf32>
      }
      scf.for %arg18 = %2 to %c1024 step %c1 {
        memref.store %cst_5, %alloc_116[%arg18] : memref<1024xf32>
      }
      cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
        %alloc_118 = memref.alloc() {alignment = 64 : i64} : memref<f32>
        linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_118 : memref<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_5 : f32
        }
        linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_116 : memref<1024xf32>) outs(%alloc_118 : memref<f32>) {
        ^bb0(%in: f32, %out: f32):
          %7 = arith.maxnumf %in, %out : f32
          linalg.yield %7 : f32
        }
        %5 = memref.load %alloc_118[] : memref<f32>
        linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%alloc_116 : memref<1024xf32>) outs(%alloc_116 : memref<1024xf32>) {
        ^bb0(%in: f32, %out: f32):
          %7 = arith.subf %in, %5 : f32
          %8 = math.exp %7 : f32
          linalg.yield %8 : f32
        }
        linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_118 : memref<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_6 : f32
        }
        linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_116 : memref<1024xf32>) outs(%alloc_118 : memref<f32>) {
        ^bb0(%in: f32, %out: f32):
          %7 = arith.addf %in, %out : f32
          linalg.yield %7 : f32
        }
        %6 = memref.load %alloc_118[] : memref<f32>
        linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%alloc_116 : memref<1024xf32>) outs(%alloc_116 : memref<1024xf32>) {
        ^bb0(%in: f32, %out: f32):
          %7 = arith.divf %in, %6 : f32
          linalg.yield %7 : f32
        }
        cinm.yield
      }
      %subview_117 = memref.subview %alloc_41[%4] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: ?>>
      linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%subview_117 : memref<48xf32, strided<[1], offset: ?>>) {
      ^bb0(%out: f32):
        linalg.yield %cst_6 : f32
      }
      scf.for %arg18 = %c0 to %2 step %c1 {
        %subview_118 = memref.subview %subview_38[%arg18, %4] [1, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<48xf32, strided<[1], offset: ?>>
        %5 = memref.load %alloc_116[%arg18] : memref<1024xf32>
        cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
          linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%subview_117, %subview_118 : memref<48xf32, strided<[1], offset: ?>>, memref<48xf32, strided<[1], offset: ?>>) outs(%subview_117 : memref<48xf32, strided<[1], offset: ?>>) {
          ^bb0(%in: f32, %in_119: f32, %out: f32):
            %6 = arith.mulf %in_119, %5 : f32
            %7 = arith.addf %in, %6 : f32
            linalg.yield %7 : f32
          }
          cinm.yield
        }
      }
    }
    memref.copy %alloc_41, %alloc_30 : memref<768xf32> to memref<768xf32>
    %subview_42 = memref.subview %arg9[0, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1]>>
    cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      memref.copy %alloc_30, %alloc_29 : memref<768xf32> to memref<768xf32>
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_42, %alloc_30 : memref<768x768xf32, strided<[768, 1]>>, memref<768xf32>) outs(%alloc_29 : memref<768xf32>) {
      ^bb0(%in: f32, %in_116: f32, %out: f32):
        %4 = arith.mulf %in, %in_116 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
      linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%subview, %alloc_29 : memref<768xf32, strided<[1], offset: ?>>, memref<768xf32>) outs(%alloc_29 : memref<768xf32>) {
      ^bb0(%in: f32, %in_116: f32, %out: f32):
        %4 = arith.addf %in, %in_116 : f32
        linalg.yield %4 : f32
      }
      cinm.yield
    }
    %subview_43 = memref.subview %arg13[0, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1]>>
    cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %alloc_116 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_116 : memref<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_6 : f32
      }
      linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_29 : memref<768xf32>) outs(%alloc_116 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %8 = arith.mulf %in, %in : f32
        %9 = arith.addf %8, %out : f32
        linalg.yield %9 : f32
      }
      %4 = memref.load %alloc_116[] : memref<f32>
      %5 = arith.divf %4, %cst_3 : f32
      %6 = arith.addf %5, %cst_2 : f32
      %7 = math.rsqrt %6 : f32
      linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%alloc_29, %subview_43 : memref<768xf32>, memref<768xf32, strided<[1]>>) outs(%alloc_40 : memref<768xf32>) {
      ^bb0(%in: f32, %in_117: f32, %out: f32):
        %8 = arith.mulf %in, %7 : f32
        %9 = arith.mulf %8, %in_117 : f32
        linalg.yield %9 : f32
      }
      cinm.yield
    }
    %subview_44 = memref.subview %arg10[0, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1]>>
    %subview_45 = memref.subview %arg12[0, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1]>>
    cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%alloc_28 : memref<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_6 : f32
      }
      memref.copy %alloc_28, %alloc_27 : memref<2048xf32> to memref<2048xf32>
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_44, %alloc_40 : memref<2048x768xf32, strided<[768, 1]>>, memref<768xf32>) outs(%alloc_27 : memref<2048xf32>) {
      ^bb0(%in: f32, %in_116: f32, %out: f32):
        %4 = arith.mulf %in, %in_116 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_45, %alloc_40 : memref<2048x768xf32, strided<[768, 1]>>, memref<768xf32>) outs(%alloc_28 : memref<2048xf32>) {
      ^bb0(%in: f32, %in_116: f32, %out: f32):
        %4 = arith.mulf %in, %in_116 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
      cinm.yield
    }
    linalg.map ins(%alloc_27, %alloc_28 : memref<2048xf32>, memref<2048xf32>) outs(%alloc_27 : memref<2048xf32>)
      (%in: f32, %in_116: f32) {
        %4 = arith.negf %in : f32
        %5 = math.exp %4 : f32
        %6 = arith.addf %5, %cst_1 : f32
        %7 = arith.divf %cst_1, %6 : f32
        %8 = arith.mulf %in, %7 : f32
        %9 = arith.mulf %8, %in_116 : f32
        linalg.yield %9 : f32
      }
    %subview_46 = memref.subview %arg11[0, 0, 0] [1, 768, 2048] [1, 1, 1] : memref<6x768x2048xf32> to memref<768x2048xf32, strided<[2048, 1]>>
    cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_46, %alloc_27 : memref<768x2048xf32, strided<[2048, 1]>>, memref<2048xf32>) outs(%alloc_40 : memref<768xf32>) {
      ^bb0(%in: f32, %in_116: f32, %out: f32):
        %4 = arith.mulf %in, %in_116 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
      cinm.yield
    }
    %subview_47 = memref.subview %arg5[1, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: 768>>
    cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %alloc_116 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_116 : memref<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_6 : f32
      }
      linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_40 : memref<768xf32>) outs(%alloc_116 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %8 = arith.mulf %in, %in : f32
        %9 = arith.addf %8, %out : f32
        linalg.yield %9 : f32
      }
      %4 = memref.load %alloc_116[] : memref<f32>
      %5 = arith.divf %4, %cst_3 : f32
      %6 = arith.addf %5, %cst_2 : f32
      %7 = math.rsqrt %6 : f32
      linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%alloc_40, %subview_47 : memref<768xf32>, memref<768xf32, strided<[1], offset: 768>>) outs(%alloc_40 : memref<768xf32>) {
      ^bb0(%in: f32, %in_117: f32, %out: f32):
        %8 = arith.mulf %in, %7 : f32
        %9 = arith.mulf %8, %in_117 : f32
        linalg.yield %9 : f32
      }
      cinm.yield
    }
    %subview_48 = memref.subview %arg6[1, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 589824>>
    %subview_49 = memref.subview %arg7[1, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 589824>>
    %subview_50 = memref.subview %arg8[1, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 589824>>
    cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%alloc_39 : memref<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_6 : f32
      }
      memref.copy %alloc_39, %alloc_26 : memref<768xf32> to memref<768xf32>
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_48, %alloc_40 : memref<768x768xf32, strided<[768, 1], offset: 589824>>, memref<768xf32>) outs(%alloc_26 : memref<768xf32>) {
      ^bb0(%in: f32, %in_119: f32, %out: f32):
        %4 = arith.mulf %in, %in_119 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
      %alloc_116 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
      memref.copy %alloc_39, %alloc_116 : memref<768xf32> to memref<768xf32>
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_49, %alloc_40 : memref<768x768xf32, strided<[768, 1], offset: 589824>>, memref<768xf32>) outs(%alloc_116 : memref<768xf32>) {
      ^bb0(%in: f32, %in_119: f32, %out: f32):
        %4 = arith.mulf %in, %in_119 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_50, %alloc_40 : memref<768x768xf32, strided<[768, 1], offset: 589824>>, memref<768xf32>) outs(%alloc_39 : memref<768xf32>) {
      ^bb0(%in: f32, %in_119: f32, %out: f32):
        %4 = arith.mulf %in, %in_119 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
      %subview_117 = memref.subview %arg2[1, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
      memref.copy %alloc_116, %subview_117 : memref<768xf32> to memref<768xf32, strided<[1], offset: ?>>
      %subview_118 = memref.subview %arg3[1, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
      memref.copy %alloc_39, %subview_118 : memref<768xf32> to memref<768xf32, strided<[1], offset: ?>>
      cinm.yield
    }
    %subview_51 = memref.subview %arg2[1, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
    scf.for %arg17 = %c0 to %c768 step %c2 {
      %4 = arith.remui %arg17, %c48 : index
      %5 = arith.index_cast %4 : index to i64
      %6 = arith.uitofp %5 : i64 to f32
      %7 = arith.divf %6, %cst_0 : f32
      %8 = math.powf %cst, %7 : f32
      %9 = arith.divf %cst_1, %8 : f32
      %10 = arith.mulf %1, %9 : f32
      %11 = math.cos %10 : f32
      %12 = math.sin %10 : f32
      %13 = arith.addi %arg17, %c1 : index
      %14 = memref.load %alloc_26[%arg17] : memref<768xf32>
      %15 = memref.load %alloc_26[%13] : memref<768xf32>
      %16 = arith.mulf %14, %11 : f32
      %17 = arith.mulf %15, %12 : f32
      %18 = arith.subf %16, %17 : f32
      memref.store %18, %alloc_26[%arg17] : memref<768xf32>
      %19 = arith.mulf %14, %12 : f32
      %20 = arith.mulf %15, %11 : f32
      %21 = arith.addf %19, %20 : f32
      memref.store %21, %alloc_26[%13] : memref<768xf32>
      %22 = arith.cmpi ult, %arg17, %c768 : index
      scf.if %22 {
        %23 = memref.load %subview_51[%arg17] : memref<768xf32, strided<[1], offset: ?>>
        %24 = memref.load %subview_51[%13] : memref<768xf32, strided<[1], offset: ?>>
        %25 = arith.mulf %23, %11 : f32
        %26 = arith.mulf %24, %12 : f32
        %27 = arith.subf %25, %26 : f32
        memref.store %27, %subview_51[%arg17] : memref<768xf32, strided<[1], offset: ?>>
        %28 = arith.mulf %23, %12 : f32
        %29 = arith.mulf %24, %11 : f32
        %30 = arith.addf %28, %29 : f32
        memref.store %30, %subview_51[%13] : memref<768xf32, strided<[1], offset: ?>>
      }
    }
    %alloc_52 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    %subview_53 = memref.subview %arg2[1, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 786432>>
    %subview_54 = memref.subview %arg3[1, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 786432>>
    %alloc_55 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc_26, %alloc_55 : memref<768xf32> to memref<768xf32>
    scf.for %arg17 = %c0 to %c6 step %c1 {
      %4 = arith.muli %arg17, %c48 : index
      %alloc_116 = memref.alloc() {alignment = 64 : i64} : memref<1024xf32>
      scf.for %arg18 = %c0 to %2 step %c1 {
        %subview_118 = memref.subview %alloc_26[%4] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: ?>>
        %subview_119 = memref.subview %subview_53[%arg18, %4] [1, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<48xf32, strided<[1], offset: ?>>
        %5 = cinm.compute -> f32 attributes {workgroupShape = array<i64: 1, 1, 8>} {
          %alloc_120 = memref.alloc() {alignment = 64 : i64} : memref<f32>
          linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_120 : memref<f32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_6 : f32
          }
          linalg.generic {indexing_maps = [#map1, #map1, #map2], iterator_types = ["reduction"]} ins(%subview_118, %subview_119 : memref<48xf32, strided<[1], offset: ?>>, memref<48xf32, strided<[1], offset: ?>>) outs(%alloc_120 : memref<f32>) {
          ^bb0(%in: f32, %in_121: f32, %out: f32):
            %8 = arith.mulf %in, %in_121 : f32
            %9 = arith.addf %8, %out : f32
            linalg.yield %9 : f32
          }
          %6 = memref.load %alloc_120[] : memref<f32>
          %7 = arith.divf %6, %cst_4 : f32
          cinm.yield %7 : f32
        }
        memref.store %5, %alloc_116[%arg18] : memref<1024xf32>
      }
      scf.for %arg18 = %2 to %c1024 step %c1 {
        memref.store %cst_5, %alloc_116[%arg18] : memref<1024xf32>
      }
      cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
        %alloc_118 = memref.alloc() {alignment = 64 : i64} : memref<f32>
        linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_118 : memref<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_5 : f32
        }
        linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_116 : memref<1024xf32>) outs(%alloc_118 : memref<f32>) {
        ^bb0(%in: f32, %out: f32):
          %7 = arith.maxnumf %in, %out : f32
          linalg.yield %7 : f32
        }
        %5 = memref.load %alloc_118[] : memref<f32>
        linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%alloc_116 : memref<1024xf32>) outs(%alloc_116 : memref<1024xf32>) {
        ^bb0(%in: f32, %out: f32):
          %7 = arith.subf %in, %5 : f32
          %8 = math.exp %7 : f32
          linalg.yield %8 : f32
        }
        linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_118 : memref<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_6 : f32
        }
        linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_116 : memref<1024xf32>) outs(%alloc_118 : memref<f32>) {
        ^bb0(%in: f32, %out: f32):
          %7 = arith.addf %in, %out : f32
          linalg.yield %7 : f32
        }
        %6 = memref.load %alloc_118[] : memref<f32>
        linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%alloc_116 : memref<1024xf32>) outs(%alloc_116 : memref<1024xf32>) {
        ^bb0(%in: f32, %out: f32):
          %7 = arith.divf %in, %6 : f32
          linalg.yield %7 : f32
        }
        cinm.yield
      }
      %subview_117 = memref.subview %alloc_55[%4] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: ?>>
      linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%subview_117 : memref<48xf32, strided<[1], offset: ?>>) {
      ^bb0(%out: f32):
        linalg.yield %cst_6 : f32
      }
      scf.for %arg18 = %c0 to %2 step %c1 {
        %subview_118 = memref.subview %subview_54[%arg18, %4] [1, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<48xf32, strided<[1], offset: ?>>
        %5 = memref.load %alloc_116[%arg18] : memref<1024xf32>
        cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
          linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%subview_117, %subview_118 : memref<48xf32, strided<[1], offset: ?>>, memref<48xf32, strided<[1], offset: ?>>) outs(%subview_117 : memref<48xf32, strided<[1], offset: ?>>) {
          ^bb0(%in: f32, %in_119: f32, %out: f32):
            %6 = arith.mulf %in_119, %5 : f32
            %7 = arith.addf %in, %6 : f32
            linalg.yield %7 : f32
          }
          cinm.yield
        }
      }
    }
    memref.copy %alloc_55, %alloc_26 : memref<768xf32> to memref<768xf32>
    %subview_56 = memref.subview %arg9[1, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 589824>>
    cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      memref.copy %alloc_26, %alloc_25 : memref<768xf32> to memref<768xf32>
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_56, %alloc_26 : memref<768x768xf32, strided<[768, 1], offset: 589824>>, memref<768xf32>) outs(%alloc_25 : memref<768xf32>) {
      ^bb0(%in: f32, %in_116: f32, %out: f32):
        %4 = arith.mulf %in, %in_116 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
      linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%alloc_40, %alloc_25 : memref<768xf32>, memref<768xf32>) outs(%alloc_25 : memref<768xf32>) {
      ^bb0(%in: f32, %in_116: f32, %out: f32):
        %4 = arith.addf %in, %in_116 : f32
        linalg.yield %4 : f32
      }
      cinm.yield
    }
    %subview_57 = memref.subview %arg13[1, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: 768>>
    cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %alloc_116 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_116 : memref<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_6 : f32
      }
      linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_25 : memref<768xf32>) outs(%alloc_116 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %8 = arith.mulf %in, %in : f32
        %9 = arith.addf %8, %out : f32
        linalg.yield %9 : f32
      }
      %4 = memref.load %alloc_116[] : memref<f32>
      %5 = arith.divf %4, %cst_3 : f32
      %6 = arith.addf %5, %cst_2 : f32
      %7 = math.rsqrt %6 : f32
      linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%alloc_25, %subview_57 : memref<768xf32>, memref<768xf32, strided<[1], offset: 768>>) outs(%alloc_40 : memref<768xf32>) {
      ^bb0(%in: f32, %in_117: f32, %out: f32):
        %8 = arith.mulf %in, %7 : f32
        %9 = arith.mulf %8, %in_117 : f32
        linalg.yield %9 : f32
      }
      cinm.yield
    }
    %subview_58 = memref.subview %arg10[1, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: 1572864>>
    %subview_59 = memref.subview %arg12[1, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: 1572864>>
    cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%alloc_24 : memref<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_6 : f32
      }
      memref.copy %alloc_24, %alloc_23 : memref<2048xf32> to memref<2048xf32>
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_58, %alloc_40 : memref<2048x768xf32, strided<[768, 1], offset: 1572864>>, memref<768xf32>) outs(%alloc_23 : memref<2048xf32>) {
      ^bb0(%in: f32, %in_116: f32, %out: f32):
        %4 = arith.mulf %in, %in_116 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_59, %alloc_40 : memref<2048x768xf32, strided<[768, 1], offset: 1572864>>, memref<768xf32>) outs(%alloc_24 : memref<2048xf32>) {
      ^bb0(%in: f32, %in_116: f32, %out: f32):
        %4 = arith.mulf %in, %in_116 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
      cinm.yield
    }
    linalg.map ins(%alloc_23, %alloc_24 : memref<2048xf32>, memref<2048xf32>) outs(%alloc_23 : memref<2048xf32>)
      (%in: f32, %in_116: f32) {
        %4 = arith.negf %in : f32
        %5 = math.exp %4 : f32
        %6 = arith.addf %5, %cst_1 : f32
        %7 = arith.divf %cst_1, %6 : f32
        %8 = arith.mulf %in, %7 : f32
        %9 = arith.mulf %8, %in_116 : f32
        linalg.yield %9 : f32
      }
    %subview_60 = memref.subview %arg11[1, 0, 0] [1, 768, 2048] [1, 1, 1] : memref<6x768x2048xf32> to memref<768x2048xf32, strided<[2048, 1], offset: 1572864>>
    cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_60, %alloc_23 : memref<768x2048xf32, strided<[2048, 1], offset: 1572864>>, memref<2048xf32>) outs(%alloc_40 : memref<768xf32>) {
      ^bb0(%in: f32, %in_116: f32, %out: f32):
        %4 = arith.mulf %in, %in_116 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
      cinm.yield
    }
    %subview_61 = memref.subview %arg5[2, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: 1536>>
    cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %alloc_116 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_116 : memref<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_6 : f32
      }
      linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_40 : memref<768xf32>) outs(%alloc_116 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %8 = arith.mulf %in, %in : f32
        %9 = arith.addf %8, %out : f32
        linalg.yield %9 : f32
      }
      %4 = memref.load %alloc_116[] : memref<f32>
      %5 = arith.divf %4, %cst_3 : f32
      %6 = arith.addf %5, %cst_2 : f32
      %7 = math.rsqrt %6 : f32
      linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%alloc_40, %subview_61 : memref<768xf32>, memref<768xf32, strided<[1], offset: 1536>>) outs(%alloc_40 : memref<768xf32>) {
      ^bb0(%in: f32, %in_117: f32, %out: f32):
        %8 = arith.mulf %in, %7 : f32
        %9 = arith.mulf %8, %in_117 : f32
        linalg.yield %9 : f32
      }
      cinm.yield
    }
    %subview_62 = memref.subview %arg6[2, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 1179648>>
    %subview_63 = memref.subview %arg7[2, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 1179648>>
    %subview_64 = memref.subview %arg8[2, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 1179648>>
    cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%alloc_52 : memref<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_6 : f32
      }
      memref.copy %alloc_52, %alloc_22 : memref<768xf32> to memref<768xf32>
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_62, %alloc_40 : memref<768x768xf32, strided<[768, 1], offset: 1179648>>, memref<768xf32>) outs(%alloc_22 : memref<768xf32>) {
      ^bb0(%in: f32, %in_119: f32, %out: f32):
        %4 = arith.mulf %in, %in_119 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
      %alloc_116 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
      memref.copy %alloc_52, %alloc_116 : memref<768xf32> to memref<768xf32>
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_63, %alloc_40 : memref<768x768xf32, strided<[768, 1], offset: 1179648>>, memref<768xf32>) outs(%alloc_116 : memref<768xf32>) {
      ^bb0(%in: f32, %in_119: f32, %out: f32):
        %4 = arith.mulf %in, %in_119 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_64, %alloc_40 : memref<768x768xf32, strided<[768, 1], offset: 1179648>>, memref<768xf32>) outs(%alloc_52 : memref<768xf32>) {
      ^bb0(%in: f32, %in_119: f32, %out: f32):
        %4 = arith.mulf %in, %in_119 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
      %subview_117 = memref.subview %arg2[2, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
      memref.copy %alloc_116, %subview_117 : memref<768xf32> to memref<768xf32, strided<[1], offset: ?>>
      %subview_118 = memref.subview %arg3[2, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
      memref.copy %alloc_52, %subview_118 : memref<768xf32> to memref<768xf32, strided<[1], offset: ?>>
      cinm.yield
    }
    %subview_65 = memref.subview %arg2[2, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
    scf.for %arg17 = %c0 to %c768 step %c2 {
      %4 = arith.remui %arg17, %c48 : index
      %5 = arith.index_cast %4 : index to i64
      %6 = arith.uitofp %5 : i64 to f32
      %7 = arith.divf %6, %cst_0 : f32
      %8 = math.powf %cst, %7 : f32
      %9 = arith.divf %cst_1, %8 : f32
      %10 = arith.mulf %1, %9 : f32
      %11 = math.cos %10 : f32
      %12 = math.sin %10 : f32
      %13 = arith.addi %arg17, %c1 : index
      %14 = memref.load %alloc_22[%arg17] : memref<768xf32>
      %15 = memref.load %alloc_22[%13] : memref<768xf32>
      %16 = arith.mulf %14, %11 : f32
      %17 = arith.mulf %15, %12 : f32
      %18 = arith.subf %16, %17 : f32
      memref.store %18, %alloc_22[%arg17] : memref<768xf32>
      %19 = arith.mulf %14, %12 : f32
      %20 = arith.mulf %15, %11 : f32
      %21 = arith.addf %19, %20 : f32
      memref.store %21, %alloc_22[%13] : memref<768xf32>
      %22 = arith.cmpi ult, %arg17, %c768 : index
      scf.if %22 {
        %23 = memref.load %subview_65[%arg17] : memref<768xf32, strided<[1], offset: ?>>
        %24 = memref.load %subview_65[%13] : memref<768xf32, strided<[1], offset: ?>>
        %25 = arith.mulf %23, %11 : f32
        %26 = arith.mulf %24, %12 : f32
        %27 = arith.subf %25, %26 : f32
        memref.store %27, %subview_65[%arg17] : memref<768xf32, strided<[1], offset: ?>>
        %28 = arith.mulf %23, %12 : f32
        %29 = arith.mulf %24, %11 : f32
        %30 = arith.addf %28, %29 : f32
        memref.store %30, %subview_65[%13] : memref<768xf32, strided<[1], offset: ?>>
      }
    }
    %alloc_66 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    %subview_67 = memref.subview %arg2[2, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 1572864>>
    %subview_68 = memref.subview %arg3[2, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 1572864>>
    %alloc_69 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc_22, %alloc_69 : memref<768xf32> to memref<768xf32>
    scf.for %arg17 = %c0 to %c6 step %c1 {
      %4 = arith.muli %arg17, %c48 : index
      %alloc_116 = memref.alloc() {alignment = 64 : i64} : memref<1024xf32>
      scf.for %arg18 = %c0 to %2 step %c1 {
        %subview_118 = memref.subview %alloc_22[%4] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: ?>>
        %subview_119 = memref.subview %subview_67[%arg18, %4] [1, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<48xf32, strided<[1], offset: ?>>
        %5 = cinm.compute -> f32 attributes {workgroupShape = array<i64: 1, 1, 8>} {
          %alloc_120 = memref.alloc() {alignment = 64 : i64} : memref<f32>
          linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_120 : memref<f32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_6 : f32
          }
          linalg.generic {indexing_maps = [#map1, #map1, #map2], iterator_types = ["reduction"]} ins(%subview_118, %subview_119 : memref<48xf32, strided<[1], offset: ?>>, memref<48xf32, strided<[1], offset: ?>>) outs(%alloc_120 : memref<f32>) {
          ^bb0(%in: f32, %in_121: f32, %out: f32):
            %8 = arith.mulf %in, %in_121 : f32
            %9 = arith.addf %8, %out : f32
            linalg.yield %9 : f32
          }
          %6 = memref.load %alloc_120[] : memref<f32>
          %7 = arith.divf %6, %cst_4 : f32
          cinm.yield %7 : f32
        }
        memref.store %5, %alloc_116[%arg18] : memref<1024xf32>
      }
      scf.for %arg18 = %2 to %c1024 step %c1 {
        memref.store %cst_5, %alloc_116[%arg18] : memref<1024xf32>
      }
      cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
        %alloc_118 = memref.alloc() {alignment = 64 : i64} : memref<f32>
        linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_118 : memref<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_5 : f32
        }
        linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_116 : memref<1024xf32>) outs(%alloc_118 : memref<f32>) {
        ^bb0(%in: f32, %out: f32):
          %7 = arith.maxnumf %in, %out : f32
          linalg.yield %7 : f32
        }
        %5 = memref.load %alloc_118[] : memref<f32>
        linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%alloc_116 : memref<1024xf32>) outs(%alloc_116 : memref<1024xf32>) {
        ^bb0(%in: f32, %out: f32):
          %7 = arith.subf %in, %5 : f32
          %8 = math.exp %7 : f32
          linalg.yield %8 : f32
        }
        linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_118 : memref<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_6 : f32
        }
        linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_116 : memref<1024xf32>) outs(%alloc_118 : memref<f32>) {
        ^bb0(%in: f32, %out: f32):
          %7 = arith.addf %in, %out : f32
          linalg.yield %7 : f32
        }
        %6 = memref.load %alloc_118[] : memref<f32>
        linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%alloc_116 : memref<1024xf32>) outs(%alloc_116 : memref<1024xf32>) {
        ^bb0(%in: f32, %out: f32):
          %7 = arith.divf %in, %6 : f32
          linalg.yield %7 : f32
        }
        cinm.yield
      }
      %subview_117 = memref.subview %alloc_69[%4] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: ?>>
      linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%subview_117 : memref<48xf32, strided<[1], offset: ?>>) {
      ^bb0(%out: f32):
        linalg.yield %cst_6 : f32
      }
      scf.for %arg18 = %c0 to %2 step %c1 {
        %subview_118 = memref.subview %subview_68[%arg18, %4] [1, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<48xf32, strided<[1], offset: ?>>
        %5 = memref.load %alloc_116[%arg18] : memref<1024xf32>
        cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
          linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%subview_117, %subview_118 : memref<48xf32, strided<[1], offset: ?>>, memref<48xf32, strided<[1], offset: ?>>) outs(%subview_117 : memref<48xf32, strided<[1], offset: ?>>) {
          ^bb0(%in: f32, %in_119: f32, %out: f32):
            %6 = arith.mulf %in_119, %5 : f32
            %7 = arith.addf %in, %6 : f32
            linalg.yield %7 : f32
          }
          cinm.yield
        }
      }
    }
    memref.copy %alloc_69, %alloc_22 : memref<768xf32> to memref<768xf32>
    %subview_70 = memref.subview %arg9[2, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 1179648>>
    cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      memref.copy %alloc_22, %alloc_21 : memref<768xf32> to memref<768xf32>
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_70, %alloc_22 : memref<768x768xf32, strided<[768, 1], offset: 1179648>>, memref<768xf32>) outs(%alloc_21 : memref<768xf32>) {
      ^bb0(%in: f32, %in_116: f32, %out: f32):
        %4 = arith.mulf %in, %in_116 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
      linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%alloc_40, %alloc_21 : memref<768xf32>, memref<768xf32>) outs(%alloc_21 : memref<768xf32>) {
      ^bb0(%in: f32, %in_116: f32, %out: f32):
        %4 = arith.addf %in, %in_116 : f32
        linalg.yield %4 : f32
      }
      cinm.yield
    }
    %subview_71 = memref.subview %arg13[2, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: 1536>>
    cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %alloc_116 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_116 : memref<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_6 : f32
      }
      linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_21 : memref<768xf32>) outs(%alloc_116 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %8 = arith.mulf %in, %in : f32
        %9 = arith.addf %8, %out : f32
        linalg.yield %9 : f32
      }
      %4 = memref.load %alloc_116[] : memref<f32>
      %5 = arith.divf %4, %cst_3 : f32
      %6 = arith.addf %5, %cst_2 : f32
      %7 = math.rsqrt %6 : f32
      linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%alloc_21, %subview_71 : memref<768xf32>, memref<768xf32, strided<[1], offset: 1536>>) outs(%alloc_40 : memref<768xf32>) {
      ^bb0(%in: f32, %in_117: f32, %out: f32):
        %8 = arith.mulf %in, %7 : f32
        %9 = arith.mulf %8, %in_117 : f32
        linalg.yield %9 : f32
      }
      cinm.yield
    }
    %subview_72 = memref.subview %arg10[2, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: 3145728>>
    %subview_73 = memref.subview %arg12[2, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: 3145728>>
    cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%alloc_20 : memref<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_6 : f32
      }
      memref.copy %alloc_20, %alloc_19 : memref<2048xf32> to memref<2048xf32>
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_72, %alloc_40 : memref<2048x768xf32, strided<[768, 1], offset: 3145728>>, memref<768xf32>) outs(%alloc_19 : memref<2048xf32>) {
      ^bb0(%in: f32, %in_116: f32, %out: f32):
        %4 = arith.mulf %in, %in_116 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_73, %alloc_40 : memref<2048x768xf32, strided<[768, 1], offset: 3145728>>, memref<768xf32>) outs(%alloc_20 : memref<2048xf32>) {
      ^bb0(%in: f32, %in_116: f32, %out: f32):
        %4 = arith.mulf %in, %in_116 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
      cinm.yield
    }
    linalg.map ins(%alloc_19, %alloc_20 : memref<2048xf32>, memref<2048xf32>) outs(%alloc_19 : memref<2048xf32>)
      (%in: f32, %in_116: f32) {
        %4 = arith.negf %in : f32
        %5 = math.exp %4 : f32
        %6 = arith.addf %5, %cst_1 : f32
        %7 = arith.divf %cst_1, %6 : f32
        %8 = arith.mulf %in, %7 : f32
        %9 = arith.mulf %8, %in_116 : f32
        linalg.yield %9 : f32
      }
    %subview_74 = memref.subview %arg11[2, 0, 0] [1, 768, 2048] [1, 1, 1] : memref<6x768x2048xf32> to memref<768x2048xf32, strided<[2048, 1], offset: 3145728>>
    cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_74, %alloc_19 : memref<768x2048xf32, strided<[2048, 1], offset: 3145728>>, memref<2048xf32>) outs(%alloc_40 : memref<768xf32>) {
      ^bb0(%in: f32, %in_116: f32, %out: f32):
        %4 = arith.mulf %in, %in_116 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
      cinm.yield
    }
    %subview_75 = memref.subview %arg5[3, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: 2304>>
    cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %alloc_116 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_116 : memref<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_6 : f32
      }
      linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_40 : memref<768xf32>) outs(%alloc_116 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %8 = arith.mulf %in, %in : f32
        %9 = arith.addf %8, %out : f32
        linalg.yield %9 : f32
      }
      %4 = memref.load %alloc_116[] : memref<f32>
      %5 = arith.divf %4, %cst_3 : f32
      %6 = arith.addf %5, %cst_2 : f32
      %7 = math.rsqrt %6 : f32
      linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%alloc_40, %subview_75 : memref<768xf32>, memref<768xf32, strided<[1], offset: 2304>>) outs(%alloc_40 : memref<768xf32>) {
      ^bb0(%in: f32, %in_117: f32, %out: f32):
        %8 = arith.mulf %in, %7 : f32
        %9 = arith.mulf %8, %in_117 : f32
        linalg.yield %9 : f32
      }
      cinm.yield
    }
    %subview_76 = memref.subview %arg6[3, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 1769472>>
    %subview_77 = memref.subview %arg7[3, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 1769472>>
    %subview_78 = memref.subview %arg8[3, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 1769472>>
    cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%alloc_66 : memref<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_6 : f32
      }
      memref.copy %alloc_66, %alloc_18 : memref<768xf32> to memref<768xf32>
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_76, %alloc_40 : memref<768x768xf32, strided<[768, 1], offset: 1769472>>, memref<768xf32>) outs(%alloc_18 : memref<768xf32>) {
      ^bb0(%in: f32, %in_119: f32, %out: f32):
        %4 = arith.mulf %in, %in_119 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
      %alloc_116 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
      memref.copy %alloc_66, %alloc_116 : memref<768xf32> to memref<768xf32>
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_77, %alloc_40 : memref<768x768xf32, strided<[768, 1], offset: 1769472>>, memref<768xf32>) outs(%alloc_116 : memref<768xf32>) {
      ^bb0(%in: f32, %in_119: f32, %out: f32):
        %4 = arith.mulf %in, %in_119 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_78, %alloc_40 : memref<768x768xf32, strided<[768, 1], offset: 1769472>>, memref<768xf32>) outs(%alloc_66 : memref<768xf32>) {
      ^bb0(%in: f32, %in_119: f32, %out: f32):
        %4 = arith.mulf %in, %in_119 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
      %subview_117 = memref.subview %arg2[3, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
      memref.copy %alloc_116, %subview_117 : memref<768xf32> to memref<768xf32, strided<[1], offset: ?>>
      %subview_118 = memref.subview %arg3[3, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
      memref.copy %alloc_66, %subview_118 : memref<768xf32> to memref<768xf32, strided<[1], offset: ?>>
      cinm.yield
    }
    %subview_79 = memref.subview %arg2[3, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
    scf.for %arg17 = %c0 to %c768 step %c2 {
      %4 = arith.remui %arg17, %c48 : index
      %5 = arith.index_cast %4 : index to i64
      %6 = arith.uitofp %5 : i64 to f32
      %7 = arith.divf %6, %cst_0 : f32
      %8 = math.powf %cst, %7 : f32
      %9 = arith.divf %cst_1, %8 : f32
      %10 = arith.mulf %1, %9 : f32
      %11 = math.cos %10 : f32
      %12 = math.sin %10 : f32
      %13 = arith.addi %arg17, %c1 : index
      %14 = memref.load %alloc_18[%arg17] : memref<768xf32>
      %15 = memref.load %alloc_18[%13] : memref<768xf32>
      %16 = arith.mulf %14, %11 : f32
      %17 = arith.mulf %15, %12 : f32
      %18 = arith.subf %16, %17 : f32
      memref.store %18, %alloc_18[%arg17] : memref<768xf32>
      %19 = arith.mulf %14, %12 : f32
      %20 = arith.mulf %15, %11 : f32
      %21 = arith.addf %19, %20 : f32
      memref.store %21, %alloc_18[%13] : memref<768xf32>
      %22 = arith.cmpi ult, %arg17, %c768 : index
      scf.if %22 {
        %23 = memref.load %subview_79[%arg17] : memref<768xf32, strided<[1], offset: ?>>
        %24 = memref.load %subview_79[%13] : memref<768xf32, strided<[1], offset: ?>>
        %25 = arith.mulf %23, %11 : f32
        %26 = arith.mulf %24, %12 : f32
        %27 = arith.subf %25, %26 : f32
        memref.store %27, %subview_79[%arg17] : memref<768xf32, strided<[1], offset: ?>>
        %28 = arith.mulf %23, %12 : f32
        %29 = arith.mulf %24, %11 : f32
        %30 = arith.addf %28, %29 : f32
        memref.store %30, %subview_79[%13] : memref<768xf32, strided<[1], offset: ?>>
      }
    }
    %alloc_80 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    %subview_81 = memref.subview %arg2[3, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 2359296>>
    %subview_82 = memref.subview %arg3[3, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 2359296>>
    %alloc_83 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc_18, %alloc_83 : memref<768xf32> to memref<768xf32>
    scf.for %arg17 = %c0 to %c6 step %c1 {
      %4 = arith.muli %arg17, %c48 : index
      %alloc_116 = memref.alloc() {alignment = 64 : i64} : memref<1024xf32>
      scf.for %arg18 = %c0 to %2 step %c1 {
        %subview_118 = memref.subview %alloc_18[%4] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: ?>>
        %subview_119 = memref.subview %subview_81[%arg18, %4] [1, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<48xf32, strided<[1], offset: ?>>
        %5 = cinm.compute -> f32 attributes {workgroupShape = array<i64: 1, 1, 8>} {
          %alloc_120 = memref.alloc() {alignment = 64 : i64} : memref<f32>
          linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_120 : memref<f32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_6 : f32
          }
          linalg.generic {indexing_maps = [#map1, #map1, #map2], iterator_types = ["reduction"]} ins(%subview_118, %subview_119 : memref<48xf32, strided<[1], offset: ?>>, memref<48xf32, strided<[1], offset: ?>>) outs(%alloc_120 : memref<f32>) {
          ^bb0(%in: f32, %in_121: f32, %out: f32):
            %8 = arith.mulf %in, %in_121 : f32
            %9 = arith.addf %8, %out : f32
            linalg.yield %9 : f32
          }
          %6 = memref.load %alloc_120[] : memref<f32>
          %7 = arith.divf %6, %cst_4 : f32
          cinm.yield %7 : f32
        }
        memref.store %5, %alloc_116[%arg18] : memref<1024xf32>
      }
      scf.for %arg18 = %2 to %c1024 step %c1 {
        memref.store %cst_5, %alloc_116[%arg18] : memref<1024xf32>
      }
      cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
        %alloc_118 = memref.alloc() {alignment = 64 : i64} : memref<f32>
        linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_118 : memref<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_5 : f32
        }
        linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_116 : memref<1024xf32>) outs(%alloc_118 : memref<f32>) {
        ^bb0(%in: f32, %out: f32):
          %7 = arith.maxnumf %in, %out : f32
          linalg.yield %7 : f32
        }
        %5 = memref.load %alloc_118[] : memref<f32>
        linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%alloc_116 : memref<1024xf32>) outs(%alloc_116 : memref<1024xf32>) {
        ^bb0(%in: f32, %out: f32):
          %7 = arith.subf %in, %5 : f32
          %8 = math.exp %7 : f32
          linalg.yield %8 : f32
        }
        linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_118 : memref<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_6 : f32
        }
        linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_116 : memref<1024xf32>) outs(%alloc_118 : memref<f32>) {
        ^bb0(%in: f32, %out: f32):
          %7 = arith.addf %in, %out : f32
          linalg.yield %7 : f32
        }
        %6 = memref.load %alloc_118[] : memref<f32>
        linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%alloc_116 : memref<1024xf32>) outs(%alloc_116 : memref<1024xf32>) {
        ^bb0(%in: f32, %out: f32):
          %7 = arith.divf %in, %6 : f32
          linalg.yield %7 : f32
        }
        cinm.yield
      }
      %subview_117 = memref.subview %alloc_83[%4] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: ?>>
      linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%subview_117 : memref<48xf32, strided<[1], offset: ?>>) {
      ^bb0(%out: f32):
        linalg.yield %cst_6 : f32
      }
      scf.for %arg18 = %c0 to %2 step %c1 {
        %subview_118 = memref.subview %subview_82[%arg18, %4] [1, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<48xf32, strided<[1], offset: ?>>
        %5 = memref.load %alloc_116[%arg18] : memref<1024xf32>
        cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
          linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%subview_117, %subview_118 : memref<48xf32, strided<[1], offset: ?>>, memref<48xf32, strided<[1], offset: ?>>) outs(%subview_117 : memref<48xf32, strided<[1], offset: ?>>) {
          ^bb0(%in: f32, %in_119: f32, %out: f32):
            %6 = arith.mulf %in_119, %5 : f32
            %7 = arith.addf %in, %6 : f32
            linalg.yield %7 : f32
          }
          cinm.yield
        }
      }
    }
    memref.copy %alloc_83, %alloc_18 : memref<768xf32> to memref<768xf32>
    %subview_84 = memref.subview %arg9[3, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 1769472>>
    cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      memref.copy %alloc_18, %alloc_17 : memref<768xf32> to memref<768xf32>
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_84, %alloc_18 : memref<768x768xf32, strided<[768, 1], offset: 1769472>>, memref<768xf32>) outs(%alloc_17 : memref<768xf32>) {
      ^bb0(%in: f32, %in_116: f32, %out: f32):
        %4 = arith.mulf %in, %in_116 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
      linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%alloc_40, %alloc_17 : memref<768xf32>, memref<768xf32>) outs(%alloc_17 : memref<768xf32>) {
      ^bb0(%in: f32, %in_116: f32, %out: f32):
        %4 = arith.addf %in, %in_116 : f32
        linalg.yield %4 : f32
      }
      cinm.yield
    }
    %subview_85 = memref.subview %arg13[3, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: 2304>>
    cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %alloc_116 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_116 : memref<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_6 : f32
      }
      linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_17 : memref<768xf32>) outs(%alloc_116 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %8 = arith.mulf %in, %in : f32
        %9 = arith.addf %8, %out : f32
        linalg.yield %9 : f32
      }
      %4 = memref.load %alloc_116[] : memref<f32>
      %5 = arith.divf %4, %cst_3 : f32
      %6 = arith.addf %5, %cst_2 : f32
      %7 = math.rsqrt %6 : f32
      linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%alloc_17, %subview_85 : memref<768xf32>, memref<768xf32, strided<[1], offset: 2304>>) outs(%alloc_40 : memref<768xf32>) {
      ^bb0(%in: f32, %in_117: f32, %out: f32):
        %8 = arith.mulf %in, %7 : f32
        %9 = arith.mulf %8, %in_117 : f32
        linalg.yield %9 : f32
      }
      cinm.yield
    }
    %subview_86 = memref.subview %arg10[3, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: 4718592>>
    %subview_87 = memref.subview %arg12[3, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: 4718592>>
    cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%alloc_16 : memref<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_6 : f32
      }
      memref.copy %alloc_16, %alloc_15 : memref<2048xf32> to memref<2048xf32>
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_86, %alloc_40 : memref<2048x768xf32, strided<[768, 1], offset: 4718592>>, memref<768xf32>) outs(%alloc_15 : memref<2048xf32>) {
      ^bb0(%in: f32, %in_116: f32, %out: f32):
        %4 = arith.mulf %in, %in_116 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_87, %alloc_40 : memref<2048x768xf32, strided<[768, 1], offset: 4718592>>, memref<768xf32>) outs(%alloc_16 : memref<2048xf32>) {
      ^bb0(%in: f32, %in_116: f32, %out: f32):
        %4 = arith.mulf %in, %in_116 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
      cinm.yield
    }
    linalg.map ins(%alloc_15, %alloc_16 : memref<2048xf32>, memref<2048xf32>) outs(%alloc_15 : memref<2048xf32>)
      (%in: f32, %in_116: f32) {
        %4 = arith.negf %in : f32
        %5 = math.exp %4 : f32
        %6 = arith.addf %5, %cst_1 : f32
        %7 = arith.divf %cst_1, %6 : f32
        %8 = arith.mulf %in, %7 : f32
        %9 = arith.mulf %8, %in_116 : f32
        linalg.yield %9 : f32
      }
    %subview_88 = memref.subview %arg11[3, 0, 0] [1, 768, 2048] [1, 1, 1] : memref<6x768x2048xf32> to memref<768x2048xf32, strided<[2048, 1], offset: 4718592>>
    cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_88, %alloc_15 : memref<768x2048xf32, strided<[2048, 1], offset: 4718592>>, memref<2048xf32>) outs(%alloc_40 : memref<768xf32>) {
      ^bb0(%in: f32, %in_116: f32, %out: f32):
        %4 = arith.mulf %in, %in_116 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
      cinm.yield
    }
    %subview_89 = memref.subview %arg5[4, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: 3072>>
    cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %alloc_116 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_116 : memref<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_6 : f32
      }
      linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_40 : memref<768xf32>) outs(%alloc_116 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %8 = arith.mulf %in, %in : f32
        %9 = arith.addf %8, %out : f32
        linalg.yield %9 : f32
      }
      %4 = memref.load %alloc_116[] : memref<f32>
      %5 = arith.divf %4, %cst_3 : f32
      %6 = arith.addf %5, %cst_2 : f32
      %7 = math.rsqrt %6 : f32
      linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%alloc_40, %subview_89 : memref<768xf32>, memref<768xf32, strided<[1], offset: 3072>>) outs(%alloc_40 : memref<768xf32>) {
      ^bb0(%in: f32, %in_117: f32, %out: f32):
        %8 = arith.mulf %in, %7 : f32
        %9 = arith.mulf %8, %in_117 : f32
        linalg.yield %9 : f32
      }
      cinm.yield
    }
    %subview_90 = memref.subview %arg6[4, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 2359296>>
    %subview_91 = memref.subview %arg7[4, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 2359296>>
    %subview_92 = memref.subview %arg8[4, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 2359296>>
    cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%alloc_80 : memref<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_6 : f32
      }
      memref.copy %alloc_80, %alloc_14 : memref<768xf32> to memref<768xf32>
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_90, %alloc_40 : memref<768x768xf32, strided<[768, 1], offset: 2359296>>, memref<768xf32>) outs(%alloc_14 : memref<768xf32>) {
      ^bb0(%in: f32, %in_119: f32, %out: f32):
        %4 = arith.mulf %in, %in_119 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
      %alloc_116 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
      memref.copy %alloc_80, %alloc_116 : memref<768xf32> to memref<768xf32>
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_91, %alloc_40 : memref<768x768xf32, strided<[768, 1], offset: 2359296>>, memref<768xf32>) outs(%alloc_116 : memref<768xf32>) {
      ^bb0(%in: f32, %in_119: f32, %out: f32):
        %4 = arith.mulf %in, %in_119 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_92, %alloc_40 : memref<768x768xf32, strided<[768, 1], offset: 2359296>>, memref<768xf32>) outs(%alloc_80 : memref<768xf32>) {
      ^bb0(%in: f32, %in_119: f32, %out: f32):
        %4 = arith.mulf %in, %in_119 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
      %subview_117 = memref.subview %arg2[4, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
      memref.copy %alloc_116, %subview_117 : memref<768xf32> to memref<768xf32, strided<[1], offset: ?>>
      %subview_118 = memref.subview %arg3[4, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
      memref.copy %alloc_80, %subview_118 : memref<768xf32> to memref<768xf32, strided<[1], offset: ?>>
      cinm.yield
    }
    %subview_93 = memref.subview %arg2[4, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
    scf.for %arg17 = %c0 to %c768 step %c2 {
      %4 = arith.remui %arg17, %c48 : index
      %5 = arith.index_cast %4 : index to i64
      %6 = arith.uitofp %5 : i64 to f32
      %7 = arith.divf %6, %cst_0 : f32
      %8 = math.powf %cst, %7 : f32
      %9 = arith.divf %cst_1, %8 : f32
      %10 = arith.mulf %1, %9 : f32
      %11 = math.cos %10 : f32
      %12 = math.sin %10 : f32
      %13 = arith.addi %arg17, %c1 : index
      %14 = memref.load %alloc_14[%arg17] : memref<768xf32>
      %15 = memref.load %alloc_14[%13] : memref<768xf32>
      %16 = arith.mulf %14, %11 : f32
      %17 = arith.mulf %15, %12 : f32
      %18 = arith.subf %16, %17 : f32
      memref.store %18, %alloc_14[%arg17] : memref<768xf32>
      %19 = arith.mulf %14, %12 : f32
      %20 = arith.mulf %15, %11 : f32
      %21 = arith.addf %19, %20 : f32
      memref.store %21, %alloc_14[%13] : memref<768xf32>
      %22 = arith.cmpi ult, %arg17, %c768 : index
      scf.if %22 {
        %23 = memref.load %subview_93[%arg17] : memref<768xf32, strided<[1], offset: ?>>
        %24 = memref.load %subview_93[%13] : memref<768xf32, strided<[1], offset: ?>>
        %25 = arith.mulf %23, %11 : f32
        %26 = arith.mulf %24, %12 : f32
        %27 = arith.subf %25, %26 : f32
        memref.store %27, %subview_93[%arg17] : memref<768xf32, strided<[1], offset: ?>>
        %28 = arith.mulf %23, %12 : f32
        %29 = arith.mulf %24, %11 : f32
        %30 = arith.addf %28, %29 : f32
        memref.store %30, %subview_93[%13] : memref<768xf32, strided<[1], offset: ?>>
      }
    }
    %alloc_94 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    %subview_95 = memref.subview %arg2[4, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3145728>>
    %subview_96 = memref.subview %arg3[4, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3145728>>
    %alloc_97 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc_14, %alloc_97 : memref<768xf32> to memref<768xf32>
    scf.for %arg17 = %c0 to %c6 step %c1 {
      %4 = arith.muli %arg17, %c48 : index
      %alloc_116 = memref.alloc() {alignment = 64 : i64} : memref<1024xf32>
      scf.for %arg18 = %c0 to %2 step %c1 {
        %subview_118 = memref.subview %alloc_14[%4] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: ?>>
        %subview_119 = memref.subview %subview_95[%arg18, %4] [1, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<48xf32, strided<[1], offset: ?>>
        %5 = cinm.compute -> f32 attributes {workgroupShape = array<i64: 1, 1, 8>} {
          %alloc_120 = memref.alloc() {alignment = 64 : i64} : memref<f32>
          linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_120 : memref<f32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_6 : f32
          }
          linalg.generic {indexing_maps = [#map1, #map1, #map2], iterator_types = ["reduction"]} ins(%subview_118, %subview_119 : memref<48xf32, strided<[1], offset: ?>>, memref<48xf32, strided<[1], offset: ?>>) outs(%alloc_120 : memref<f32>) {
          ^bb0(%in: f32, %in_121: f32, %out: f32):
            %8 = arith.mulf %in, %in_121 : f32
            %9 = arith.addf %8, %out : f32
            linalg.yield %9 : f32
          }
          %6 = memref.load %alloc_120[] : memref<f32>
          %7 = arith.divf %6, %cst_4 : f32
          cinm.yield %7 : f32
        }
        memref.store %5, %alloc_116[%arg18] : memref<1024xf32>
      }
      scf.for %arg18 = %2 to %c1024 step %c1 {
        memref.store %cst_5, %alloc_116[%arg18] : memref<1024xf32>
      }
      cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
        %alloc_118 = memref.alloc() {alignment = 64 : i64} : memref<f32>
        linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_118 : memref<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_5 : f32
        }
        linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_116 : memref<1024xf32>) outs(%alloc_118 : memref<f32>) {
        ^bb0(%in: f32, %out: f32):
          %7 = arith.maxnumf %in, %out : f32
          linalg.yield %7 : f32
        }
        %5 = memref.load %alloc_118[] : memref<f32>
        linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%alloc_116 : memref<1024xf32>) outs(%alloc_116 : memref<1024xf32>) {
        ^bb0(%in: f32, %out: f32):
          %7 = arith.subf %in, %5 : f32
          %8 = math.exp %7 : f32
          linalg.yield %8 : f32
        }
        linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_118 : memref<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_6 : f32
        }
        linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_116 : memref<1024xf32>) outs(%alloc_118 : memref<f32>) {
        ^bb0(%in: f32, %out: f32):
          %7 = arith.addf %in, %out : f32
          linalg.yield %7 : f32
        }
        %6 = memref.load %alloc_118[] : memref<f32>
        linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%alloc_116 : memref<1024xf32>) outs(%alloc_116 : memref<1024xf32>) {
        ^bb0(%in: f32, %out: f32):
          %7 = arith.divf %in, %6 : f32
          linalg.yield %7 : f32
        }
        cinm.yield
      }
      %subview_117 = memref.subview %alloc_97[%4] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: ?>>
      linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%subview_117 : memref<48xf32, strided<[1], offset: ?>>) {
      ^bb0(%out: f32):
        linalg.yield %cst_6 : f32
      }
      scf.for %arg18 = %c0 to %2 step %c1 {
        %subview_118 = memref.subview %subview_96[%arg18, %4] [1, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<48xf32, strided<[1], offset: ?>>
        %5 = memref.load %alloc_116[%arg18] : memref<1024xf32>
        cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
          linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%subview_117, %subview_118 : memref<48xf32, strided<[1], offset: ?>>, memref<48xf32, strided<[1], offset: ?>>) outs(%subview_117 : memref<48xf32, strided<[1], offset: ?>>) {
          ^bb0(%in: f32, %in_119: f32, %out: f32):
            %6 = arith.mulf %in_119, %5 : f32
            %7 = arith.addf %in, %6 : f32
            linalg.yield %7 : f32
          }
          cinm.yield
        }
      }
    }
    memref.copy %alloc_97, %alloc_14 : memref<768xf32> to memref<768xf32>
    %subview_98 = memref.subview %arg9[4, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 2359296>>
    cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      memref.copy %alloc_14, %alloc_13 : memref<768xf32> to memref<768xf32>
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_98, %alloc_14 : memref<768x768xf32, strided<[768, 1], offset: 2359296>>, memref<768xf32>) outs(%alloc_13 : memref<768xf32>) {
      ^bb0(%in: f32, %in_116: f32, %out: f32):
        %4 = arith.mulf %in, %in_116 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
      linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%alloc_40, %alloc_13 : memref<768xf32>, memref<768xf32>) outs(%alloc_13 : memref<768xf32>) {
      ^bb0(%in: f32, %in_116: f32, %out: f32):
        %4 = arith.addf %in, %in_116 : f32
        linalg.yield %4 : f32
      }
      cinm.yield
    }
    %subview_99 = memref.subview %arg13[4, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: 3072>>
    cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %alloc_116 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_116 : memref<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_6 : f32
      }
      linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_13 : memref<768xf32>) outs(%alloc_116 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %8 = arith.mulf %in, %in : f32
        %9 = arith.addf %8, %out : f32
        linalg.yield %9 : f32
      }
      %4 = memref.load %alloc_116[] : memref<f32>
      %5 = arith.divf %4, %cst_3 : f32
      %6 = arith.addf %5, %cst_2 : f32
      %7 = math.rsqrt %6 : f32
      linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%alloc_13, %subview_99 : memref<768xf32>, memref<768xf32, strided<[1], offset: 3072>>) outs(%alloc_40 : memref<768xf32>) {
      ^bb0(%in: f32, %in_117: f32, %out: f32):
        %8 = arith.mulf %in, %7 : f32
        %9 = arith.mulf %8, %in_117 : f32
        linalg.yield %9 : f32
      }
      cinm.yield
    }
    %subview_100 = memref.subview %arg10[4, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: 6291456>>
    %subview_101 = memref.subview %arg12[4, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: 6291456>>
    cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%alloc_12 : memref<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_6 : f32
      }
      memref.copy %alloc_12, %alloc_11 : memref<2048xf32> to memref<2048xf32>
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %alloc_40 : memref<2048x768xf32, strided<[768, 1], offset: 6291456>>, memref<768xf32>) outs(%alloc_11 : memref<2048xf32>) {
      ^bb0(%in: f32, %in_116: f32, %out: f32):
        %4 = arith.mulf %in, %in_116 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_101, %alloc_40 : memref<2048x768xf32, strided<[768, 1], offset: 6291456>>, memref<768xf32>) outs(%alloc_12 : memref<2048xf32>) {
      ^bb0(%in: f32, %in_116: f32, %out: f32):
        %4 = arith.mulf %in, %in_116 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
      cinm.yield
    }
    linalg.map ins(%alloc_11, %alloc_12 : memref<2048xf32>, memref<2048xf32>) outs(%alloc_11 : memref<2048xf32>)
      (%in: f32, %in_116: f32) {
        %4 = arith.negf %in : f32
        %5 = math.exp %4 : f32
        %6 = arith.addf %5, %cst_1 : f32
        %7 = arith.divf %cst_1, %6 : f32
        %8 = arith.mulf %in, %7 : f32
        %9 = arith.mulf %8, %in_116 : f32
        linalg.yield %9 : f32
      }
    %subview_102 = memref.subview %arg11[4, 0, 0] [1, 768, 2048] [1, 1, 1] : memref<6x768x2048xf32> to memref<768x2048xf32, strided<[2048, 1], offset: 6291456>>
    cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_102, %alloc_11 : memref<768x2048xf32, strided<[2048, 1], offset: 6291456>>, memref<2048xf32>) outs(%alloc_40 : memref<768xf32>) {
      ^bb0(%in: f32, %in_116: f32, %out: f32):
        %4 = arith.mulf %in, %in_116 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
      cinm.yield
    }
    %subview_103 = memref.subview %arg5[5, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: 3840>>
    cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %alloc_116 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_116 : memref<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_6 : f32
      }
      linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_40 : memref<768xf32>) outs(%alloc_116 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %8 = arith.mulf %in, %in : f32
        %9 = arith.addf %8, %out : f32
        linalg.yield %9 : f32
      }
      %4 = memref.load %alloc_116[] : memref<f32>
      %5 = arith.divf %4, %cst_3 : f32
      %6 = arith.addf %5, %cst_2 : f32
      %7 = math.rsqrt %6 : f32
      linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%alloc_40, %subview_103 : memref<768xf32>, memref<768xf32, strided<[1], offset: 3840>>) outs(%alloc_40 : memref<768xf32>) {
      ^bb0(%in: f32, %in_117: f32, %out: f32):
        %8 = arith.mulf %in, %7 : f32
        %9 = arith.mulf %8, %in_117 : f32
        linalg.yield %9 : f32
      }
      cinm.yield
    }
    %subview_104 = memref.subview %arg6[5, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 2949120>>
    %subview_105 = memref.subview %arg7[5, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 2949120>>
    %subview_106 = memref.subview %arg8[5, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 2949120>>
    cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%alloc_94 : memref<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_6 : f32
      }
      memref.copy %alloc_94, %alloc_10 : memref<768xf32> to memref<768xf32>
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_104, %alloc_40 : memref<768x768xf32, strided<[768, 1], offset: 2949120>>, memref<768xf32>) outs(%alloc_10 : memref<768xf32>) {
      ^bb0(%in: f32, %in_119: f32, %out: f32):
        %4 = arith.mulf %in, %in_119 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
      %alloc_116 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
      memref.copy %alloc_94, %alloc_116 : memref<768xf32> to memref<768xf32>
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_105, %alloc_40 : memref<768x768xf32, strided<[768, 1], offset: 2949120>>, memref<768xf32>) outs(%alloc_116 : memref<768xf32>) {
      ^bb0(%in: f32, %in_119: f32, %out: f32):
        %4 = arith.mulf %in, %in_119 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_106, %alloc_40 : memref<768x768xf32, strided<[768, 1], offset: 2949120>>, memref<768xf32>) outs(%alloc_94 : memref<768xf32>) {
      ^bb0(%in: f32, %in_119: f32, %out: f32):
        %4 = arith.mulf %in, %in_119 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
      %subview_117 = memref.subview %arg2[5, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
      memref.copy %alloc_116, %subview_117 : memref<768xf32> to memref<768xf32, strided<[1], offset: ?>>
      %subview_118 = memref.subview %arg3[5, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
      memref.copy %alloc_94, %subview_118 : memref<768xf32> to memref<768xf32, strided<[1], offset: ?>>
      cinm.yield
    }
    %subview_107 = memref.subview %arg2[5, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
    scf.for %arg17 = %c0 to %c768 step %c2 {
      %4 = arith.remui %arg17, %c48 : index
      %5 = arith.index_cast %4 : index to i64
      %6 = arith.uitofp %5 : i64 to f32
      %7 = arith.divf %6, %cst_0 : f32
      %8 = math.powf %cst, %7 : f32
      %9 = arith.divf %cst_1, %8 : f32
      %10 = arith.mulf %1, %9 : f32
      %11 = math.cos %10 : f32
      %12 = math.sin %10 : f32
      %13 = arith.addi %arg17, %c1 : index
      %14 = memref.load %alloc_10[%arg17] : memref<768xf32>
      %15 = memref.load %alloc_10[%13] : memref<768xf32>
      %16 = arith.mulf %14, %11 : f32
      %17 = arith.mulf %15, %12 : f32
      %18 = arith.subf %16, %17 : f32
      memref.store %18, %alloc_10[%arg17] : memref<768xf32>
      %19 = arith.mulf %14, %12 : f32
      %20 = arith.mulf %15, %11 : f32
      %21 = arith.addf %19, %20 : f32
      memref.store %21, %alloc_10[%13] : memref<768xf32>
      %22 = arith.cmpi ult, %arg17, %c768 : index
      scf.if %22 {
        %23 = memref.load %subview_107[%arg17] : memref<768xf32, strided<[1], offset: ?>>
        %24 = memref.load %subview_107[%13] : memref<768xf32, strided<[1], offset: ?>>
        %25 = arith.mulf %23, %11 : f32
        %26 = arith.mulf %24, %12 : f32
        %27 = arith.subf %25, %26 : f32
        memref.store %27, %subview_107[%arg17] : memref<768xf32, strided<[1], offset: ?>>
        %28 = arith.mulf %23, %12 : f32
        %29 = arith.mulf %24, %11 : f32
        %30 = arith.addf %28, %29 : f32
        memref.store %30, %subview_107[%13] : memref<768xf32, strided<[1], offset: ?>>
      }
    }
    %subview_108 = memref.subview %arg2[5, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3932160>>
    %subview_109 = memref.subview %arg3[5, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3932160>>
    %alloc_110 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc_10, %alloc_110 : memref<768xf32> to memref<768xf32>
    scf.for %arg17 = %c0 to %c6 step %c1 {
      %4 = arith.muli %arg17, %c48 : index
      %alloc_116 = memref.alloc() {alignment = 64 : i64} : memref<1024xf32>
      scf.for %arg18 = %c0 to %2 step %c1 {
        %subview_118 = memref.subview %alloc_10[%4] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: ?>>
        %subview_119 = memref.subview %subview_108[%arg18, %4] [1, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<48xf32, strided<[1], offset: ?>>
        %5 = cinm.compute -> f32 attributes {workgroupShape = array<i64: 1, 1, 8>} {
          %alloc_120 = memref.alloc() {alignment = 64 : i64} : memref<f32>
          linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_120 : memref<f32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_6 : f32
          }
          linalg.generic {indexing_maps = [#map1, #map1, #map2], iterator_types = ["reduction"]} ins(%subview_118, %subview_119 : memref<48xf32, strided<[1], offset: ?>>, memref<48xf32, strided<[1], offset: ?>>) outs(%alloc_120 : memref<f32>) {
          ^bb0(%in: f32, %in_121: f32, %out: f32):
            %8 = arith.mulf %in, %in_121 : f32
            %9 = arith.addf %8, %out : f32
            linalg.yield %9 : f32
          }
          %6 = memref.load %alloc_120[] : memref<f32>
          %7 = arith.divf %6, %cst_4 : f32
          cinm.yield %7 : f32
        }
        memref.store %5, %alloc_116[%arg18] : memref<1024xf32>
      }
      scf.for %arg18 = %2 to %c1024 step %c1 {
        memref.store %cst_5, %alloc_116[%arg18] : memref<1024xf32>
      }
      cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
        %alloc_118 = memref.alloc() {alignment = 64 : i64} : memref<f32>
        linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_118 : memref<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_5 : f32
        }
        linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_116 : memref<1024xf32>) outs(%alloc_118 : memref<f32>) {
        ^bb0(%in: f32, %out: f32):
          %7 = arith.maxnumf %in, %out : f32
          linalg.yield %7 : f32
        }
        %5 = memref.load %alloc_118[] : memref<f32>
        linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%alloc_116 : memref<1024xf32>) outs(%alloc_116 : memref<1024xf32>) {
        ^bb0(%in: f32, %out: f32):
          %7 = arith.subf %in, %5 : f32
          %8 = math.exp %7 : f32
          linalg.yield %8 : f32
        }
        linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_118 : memref<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_6 : f32
        }
        linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_116 : memref<1024xf32>) outs(%alloc_118 : memref<f32>) {
        ^bb0(%in: f32, %out: f32):
          %7 = arith.addf %in, %out : f32
          linalg.yield %7 : f32
        }
        %6 = memref.load %alloc_118[] : memref<f32>
        linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%alloc_116 : memref<1024xf32>) outs(%alloc_116 : memref<1024xf32>) {
        ^bb0(%in: f32, %out: f32):
          %7 = arith.divf %in, %6 : f32
          linalg.yield %7 : f32
        }
        cinm.yield
      }
      %subview_117 = memref.subview %alloc_110[%4] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: ?>>
      linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%subview_117 : memref<48xf32, strided<[1], offset: ?>>) {
      ^bb0(%out: f32):
        linalg.yield %cst_6 : f32
      }
      scf.for %arg18 = %c0 to %2 step %c1 {
        %subview_118 = memref.subview %subview_109[%arg18, %4] [1, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<48xf32, strided<[1], offset: ?>>
        %5 = memref.load %alloc_116[%arg18] : memref<1024xf32>
        cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
          linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%subview_117, %subview_118 : memref<48xf32, strided<[1], offset: ?>>, memref<48xf32, strided<[1], offset: ?>>) outs(%subview_117 : memref<48xf32, strided<[1], offset: ?>>) {
          ^bb0(%in: f32, %in_119: f32, %out: f32):
            %6 = arith.mulf %in_119, %5 : f32
            %7 = arith.addf %in, %6 : f32
            linalg.yield %7 : f32
          }
          cinm.yield
        }
      }
    }
    memref.copy %alloc_110, %alloc_10 : memref<768xf32> to memref<768xf32>
    %subview_111 = memref.subview %arg9[5, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 2949120>>
    cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      memref.copy %alloc_10, %alloc_9 : memref<768xf32> to memref<768xf32>
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_111, %alloc_10 : memref<768x768xf32, strided<[768, 1], offset: 2949120>>, memref<768xf32>) outs(%alloc_9 : memref<768xf32>) {
      ^bb0(%in: f32, %in_116: f32, %out: f32):
        %4 = arith.mulf %in, %in_116 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
      linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%alloc_40, %alloc_9 : memref<768xf32>, memref<768xf32>) outs(%alloc_9 : memref<768xf32>) {
      ^bb0(%in: f32, %in_116: f32, %out: f32):
        %4 = arith.addf %in, %in_116 : f32
        linalg.yield %4 : f32
      }
      cinm.yield
    }
    %subview_112 = memref.subview %arg13[5, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: 3840>>
    cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %alloc_116 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_116 : memref<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_6 : f32
      }
      linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_9 : memref<768xf32>) outs(%alloc_116 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %8 = arith.mulf %in, %in : f32
        %9 = arith.addf %8, %out : f32
        linalg.yield %9 : f32
      }
      %4 = memref.load %alloc_116[] : memref<f32>
      %5 = arith.divf %4, %cst_3 : f32
      %6 = arith.addf %5, %cst_2 : f32
      %7 = math.rsqrt %6 : f32
      linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%alloc_9, %subview_112 : memref<768xf32>, memref<768xf32, strided<[1], offset: 3840>>) outs(%alloc_40 : memref<768xf32>) {
      ^bb0(%in: f32, %in_117: f32, %out: f32):
        %8 = arith.mulf %in, %7 : f32
        %9 = arith.mulf %8, %in_117 : f32
        linalg.yield %9 : f32
      }
      cinm.yield
    }
    %subview_113 = memref.subview %arg10[5, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: 7864320>>
    %subview_114 = memref.subview %arg12[5, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: 7864320>>
    cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%alloc_8 : memref<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_6 : f32
      }
      memref.copy %alloc_8, %alloc_7 : memref<2048xf32> to memref<2048xf32>
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_113, %alloc_40 : memref<2048x768xf32, strided<[768, 1], offset: 7864320>>, memref<768xf32>) outs(%alloc_7 : memref<2048xf32>) {
      ^bb0(%in: f32, %in_116: f32, %out: f32):
        %4 = arith.mulf %in, %in_116 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_114, %alloc_40 : memref<2048x768xf32, strided<[768, 1], offset: 7864320>>, memref<768xf32>) outs(%alloc_8 : memref<2048xf32>) {
      ^bb0(%in: f32, %in_116: f32, %out: f32):
        %4 = arith.mulf %in, %in_116 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
      cinm.yield
    }
    linalg.map ins(%alloc_7, %alloc_8 : memref<2048xf32>, memref<2048xf32>) outs(%alloc_7 : memref<2048xf32>)
      (%in: f32, %in_116: f32) {
        %4 = arith.negf %in : f32
        %5 = math.exp %4 : f32
        %6 = arith.addf %5, %cst_1 : f32
        %7 = arith.divf %cst_1, %6 : f32
        %8 = arith.mulf %in, %7 : f32
        %9 = arith.mulf %8, %in_116 : f32
        linalg.yield %9 : f32
      }
    %subview_115 = memref.subview %arg11[5, 0, 0] [1, 768, 2048] [1, 1, 1] : memref<6x768x2048xf32> to memref<768x2048xf32, strided<[2048, 1], offset: 7864320>>
    cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_115, %alloc_7 : memref<768x2048xf32, strided<[2048, 1], offset: 7864320>>, memref<2048xf32>) outs(%alloc_40 : memref<768xf32>) {
      ^bb0(%in: f32, %in_116: f32, %out: f32):
        %4 = arith.mulf %in, %in_116 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
      cinm.yield
    }
    cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %alloc_116 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_116 : memref<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_6 : f32
      }
      linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_40 : memref<768xf32>) outs(%alloc_116 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %8 = arith.mulf %in, %in : f32
        %9 = arith.addf %8, %out : f32
        linalg.yield %9 : f32
      }
      %4 = memref.load %alloc_116[] : memref<f32>
      %5 = arith.divf %4, %cst_3 : f32
      %6 = arith.addf %5, %cst_2 : f32
      %7 = math.rsqrt %6 : f32
      linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%alloc_40, %arg14 : memref<768xf32>, memref<768xf32>) outs(%alloc_40 : memref<768xf32>) {
      ^bb0(%in: f32, %in_117: f32, %out: f32):
        %8 = arith.mulf %in, %7 : f32
        %9 = arith.mulf %8, %in_117 : f32
        linalg.yield %9 : f32
      }
      cinm.yield
    }
    %3 = cinm.compute -> memref<32000xf32, strided<[1]>> attributes {workgroupShape = array<i64: 2, 8, 16>} {
      %alloc_116 = memref.alloc() {alignment = 64 : i64} : memref<34048x768xf32>
      linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%alloc_116 : memref<34048x768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_6 : f32
      }
      %subview_117 = memref.subview %alloc_116[0, 0] [32000, 768] [1, 1] : memref<34048x768xf32> to memref<32000x768xf32, strided<[768, 1]>>
      memref.copy %arg15, %subview_117 : memref<32000x768xf32> to memref<32000x768xf32, strided<[768, 1]>>
      linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%alloc : memref<34048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_6 : f32
      }
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%alloc_116, %alloc_40 : memref<34048x768xf32>, memref<768xf32>) outs(%alloc : memref<34048xf32>) {
      ^bb0(%in: f32, %in_119: f32, %out: f32):
        %4 = arith.mulf %in, %in_119 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      }
      %subview_118 = memref.subview %alloc[0] [32000] [1] : memref<34048xf32> to memref<32000xf32, strided<[1]>>
      cinm.yield %subview_118 : memref<32000xf32, strided<[1]>>
    }
    memref.copy %3, %arg16 : memref<32000xf32, strided<[1]>> to memref<32000xf32, strided<[1]>>
    return
  }
  func.func @rot(%arg0: memref<768xf32, strided<[?], offset: ?>>, %arg1: index, %arg2: f32, %arg3: f32, %arg4: memref<768xf32, strided<[?], offset: ?>>) {
    %c1 = arith.constant 1 : index
    %0 = arith.addi %arg1, %c1 : index
    %1 = memref.load %arg0[%arg1] : memref<768xf32, strided<[?], offset: ?>>
    %2 = memref.load %arg0[%0] : memref<768xf32, strided<[?], offset: ?>>
    %3 = arith.mulf %1, %arg2 : f32
    %4 = arith.mulf %2, %arg3 : f32
    %5 = arith.subf %3, %4 : f32
    memref.store %5, %arg0[%arg1] : memref<768xf32, strided<[?], offset: ?>>
    %6 = arith.mulf %1, %arg3 : f32
    %7 = arith.mulf %2, %arg2 : f32
    %8 = arith.addf %6, %7 : f32
    memref.store %8, %arg0[%0] : memref<768xf32, strided<[?], offset: ?>>
    memref.copy %arg0, %arg4 : memref<768xf32, strided<[?], offset: ?>> to memref<768xf32, strided<[?], offset: ?>>
    return
  }
  func.func @mha(%arg0: memref<768xf32, strided<[?], offset: ?>>, %arg1: memref<1024x768xf32, strided<[?, ?], offset: ?>>, %arg2: memref<1024x768xf32, strided<[?, ?], offset: ?>>, %arg3: index, %arg4: memref<768xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    %c48 = arith.constant 48 : index
    %c1024 = arith.constant 1024 : index
    %cst_0 = arith.constant 6.92820311 : f32
    %cst_1 = arith.constant 0xFF800000 : f32
    %0 = arith.addi %arg3, %c1 : index
    scf.for %arg5 = %c0 to %c6 step %c1 {
      %1 = arith.muli %arg5, %c48 : index
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<1024xf32>
      scf.for %arg6 = %c0 to %0 step %c1 {
        %subview_2 = memref.subview %arg0[%1] [48] [1] : memref<768xf32, strided<[?], offset: ?>> to memref<48xf32, strided<[?], offset: ?>>
        %subview_3 = memref.subview %arg1[%arg6, %1] [1, 48] [1, 1] : memref<1024x768xf32, strided<[?, ?], offset: ?>> to memref<48xf32, strided<[?], offset: ?>>
        %2 = cinm.compute -> f32 attributes {workgroupShape = array<i64: 1, 1, 8>} {
          %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<f32>
          linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_4 : memref<f32>) {
          ^bb0(%out: f32):
            linalg.yield %cst : f32
          }
          linalg.generic {indexing_maps = [#map1, #map1, #map2], iterator_types = ["reduction"]} ins(%subview_2, %subview_3 : memref<48xf32, strided<[?], offset: ?>>, memref<48xf32, strided<[?], offset: ?>>) outs(%alloc_4 : memref<f32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %5 = arith.mulf %in, %in_5 : f32
            %6 = arith.addf %5, %out : f32
            linalg.yield %6 : f32
          }
          %3 = memref.load %alloc_4[] : memref<f32>
          %4 = arith.divf %3, %cst_0 : f32
          cinm.yield %4 : f32
        }
        memref.store %2, %alloc[%arg6] : memref<1024xf32>
      }
      scf.for %arg6 = %0 to %c1024 step %c1 {
        memref.store %cst_1, %alloc[%arg6] : memref<1024xf32>
      }
      cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
        %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<f32>
        linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_2 : memref<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_1 : f32
        }
        linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc : memref<1024xf32>) outs(%alloc_2 : memref<f32>) {
        ^bb0(%in: f32, %out: f32):
          %4 = arith.maxnumf %in, %out : f32
          linalg.yield %4 : f32
        }
        %2 = memref.load %alloc_2[] : memref<f32>
        linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%alloc : memref<1024xf32>) outs(%alloc : memref<1024xf32>) {
        ^bb0(%in: f32, %out: f32):
          %4 = arith.subf %in, %2 : f32
          %5 = math.exp %4 : f32
          linalg.yield %5 : f32
        }
        linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_2 : memref<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst : f32
        }
        linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc : memref<1024xf32>) outs(%alloc_2 : memref<f32>) {
        ^bb0(%in: f32, %out: f32):
          %4 = arith.addf %in, %out : f32
          linalg.yield %4 : f32
        }
        %3 = memref.load %alloc_2[] : memref<f32>
        linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%alloc : memref<1024xf32>) outs(%alloc : memref<1024xf32>) {
        ^bb0(%in: f32, %out: f32):
          %4 = arith.divf %in, %3 : f32
          linalg.yield %4 : f32
        }
        cinm.yield
      }
      %subview = memref.subview %arg4[%1] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: ?>>
      linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%subview : memref<48xf32, strided<[1], offset: ?>>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      }
      scf.for %arg6 = %c0 to %0 step %c1 {
        %subview_2 = memref.subview %arg2[%arg6, %1] [1, 48] [1, 1] : memref<1024x768xf32, strided<[?, ?], offset: ?>> to memref<48xf32, strided<[?], offset: ?>>
        %2 = memref.load %alloc[%arg6] : memref<1024xf32>
        cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
          linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%subview, %subview_2 : memref<48xf32, strided<[1], offset: ?>>, memref<48xf32, strided<[?], offset: ?>>) outs(%subview : memref<48xf32, strided<[1], offset: ?>>) {
          ^bb0(%in: f32, %in_3: f32, %out: f32):
            %3 = arith.mulf %in_3, %2 : f32
            %4 = arith.addf %in, %3 : f32
            linalg.yield %4 : f32
          }
          cinm.yield
        }
      }
    }
    return
  }
  func.func @rmsnorm(%arg0: memref<768xf32, strided<[?], offset: ?>>, %arg1: memref<768xf32, strided<[?], offset: ?>>, %arg2: memref<768xf32>) {
    %cst = arith.constant 7.680000e+02 : f32
    %cst_0 = arith.constant 9.99999974E-6 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<f32>
      linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc : memref<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_1 : f32
      }
      linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%arg0 : memref<768xf32, strided<[?], offset: ?>>) outs(%alloc : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %4 = arith.mulf %in, %in : f32
        %5 = arith.addf %4, %out : f32
        linalg.yield %5 : f32
      }
      %0 = memref.load %alloc[] : memref<f32>
      %1 = arith.divf %0, %cst : f32
      %2 = arith.addf %1, %cst_0 : f32
      %3 = math.rsqrt %2 : f32
      linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<768xf32, strided<[?], offset: ?>>, memref<768xf32, strided<[?], offset: ?>>) outs(%arg2 : memref<768xf32>) {
      ^bb0(%in: f32, %in_2: f32, %out: f32):
        %4 = arith.mulf %in, %3 : f32
        %5 = arith.mulf %4, %in_2 : f32
        linalg.yield %5 : f32
      }
      cinm.yield
    }
    return
  }
  func.func @softmax(%arg0: memref<1024xf32, strided<[?], offset: ?>>, %arg1: memref<1024xf32, strided<[?], offset: ?>>) {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    cinm.compute attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<f32>
      linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc : memref<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_0 : f32
      }
      linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%arg0 : memref<1024xf32, strided<[?], offset: ?>>) outs(%alloc : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %2 = arith.maxnumf %in, %out : f32
        linalg.yield %2 : f32
      }
      %0 = memref.load %alloc[] : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%arg0 : memref<1024xf32, strided<[?], offset: ?>>) outs(%arg0 : memref<1024xf32, strided<[?], offset: ?>>) {
      ^bb0(%in: f32, %out: f32):
        %2 = arith.subf %in, %0 : f32
        %3 = math.exp %2 : f32
        linalg.yield %3 : f32
      }
      linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc : memref<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      }
      linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%arg0 : memref<1024xf32, strided<[?], offset: ?>>) outs(%alloc : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %2 = arith.addf %in, %out : f32
        linalg.yield %2 : f32
      }
      %1 = memref.load %alloc[] : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%arg0 : memref<1024xf32, strided<[?], offset: ?>>) outs(%arg0 : memref<1024xf32, strided<[?], offset: ?>>) {
      ^bb0(%in: f32, %out: f32):
        %2 = arith.divf %in, %1 : f32
        linalg.yield %2 : f32
      }
      cinm.yield
    }
    memref.copy %arg0, %arg1 : memref<1024xf32, strided<[?], offset: ?>> to memref<1024xf32, strided<[?], offset: ?>>
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["func.func"]} attributes {sym_name = "forward"} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.structured.match ops{["scf.for"]} in %0 : (!transform.any_op) -> !transform.any_op
      %2:2 = transform.split_handle %1 {overflow_result = 1 : i64} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.loop.unroll %2#1 {factor = 6 : i64} : !transform.any_op
      transform.yield 
    }
  }
}

