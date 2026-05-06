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
    %c6 = arith.constant 6 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst_2 = arith.constant 9.99999974E-6 : f32
    %cst_3 = arith.constant 7.680000e+02 : f32
    %c1024 = arith.constant 1024 : index
    %cst_4 = arith.constant 6.92820311 : f32
    %cst_5 = arith.constant 0xFF800000 : f32
    %cst_6 = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<34048xf32>
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    %subview = memref.subview %arg4[%arg0, 0] [1, 768] [1, 1] : memref<32000x768xf32> to memref<768xf32, strided<[1], offset: ?>>
    %cast = memref.cast %subview : memref<768xf32, strided<[1], offset: ?>> to memref<768xf32, strided<[?], offset: ?>>
    %0 = scf.for %arg17 = %c0 to %c6 step %c1 iter_args(%arg18 = %cast) -> (memref<768xf32, strided<[?], offset: ?>>) {
      %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<2048xf32>
      %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<2048xf32>
      %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
      %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
      %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
      %subview_13 = memref.subview %arg5[%arg17, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: ?>>
      cinm.compute_
           attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
        %alloc_28 = memref.alloc() {alignment = 64 : i64} : memref<f32>
        linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_28 : memref<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_6 : f32
        }
        linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%arg18 : memref<768xf32, strided<[?], offset: ?>>) outs(%alloc_28 : memref<f32>) {
        ^bb0(%in: f32, %out: f32):
          %9 = arith.mulf %in, %in : f32
          %10 = arith.addf %9, %out : f32
          linalg.yield %10 : f32
        }
        %5 = memref.load %alloc_28[] : memref<f32>
        %6 = arith.divf %5, %cst_3 : f32
        %7 = arith.addf %6, %cst_2 : f32
        %8 = math.rsqrt %7 : f32
        linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%arg18, %subview_13 : memref<768xf32, strided<[?], offset: ?>>, memref<768xf32, strided<[1], offset: ?>>) outs(%alloc_12 : memref<768xf32>) {
        ^bb0(%in: f32, %in_29: f32, %out: f32):
          %9 = arith.mulf %in, %8 : f32
          %10 = arith.mulf %9, %in_29 : f32
          linalg.yield %10 : f32
        }
        cinm.yield
      }
      %subview_14 = memref.subview %arg6[%arg17, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: ?>>
      %subview_15 = memref.subview %arg7[%arg17, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: ?>>
      %subview_16 = memref.subview %arg8[%arg17, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: ?>>
      cinm.compute_
           attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
        %alloc_28 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
        linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%alloc_28 : memref<768xf32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_6 : f32
        }
        memref.copy %alloc_28, %alloc_11 : memref<768xf32> to memref<768xf32>
        linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_14, %alloc_12 : memref<768x768xf32, strided<[768, 1], offset: ?>>, memref<768xf32>) outs(%alloc_11 : memref<768xf32>) {
        ^bb0(%in: f32, %in_32: f32, %out: f32):
          %5 = arith.mulf %in, %in_32 : f32
          %6 = arith.addf %out, %5 : f32
          linalg.yield %6 : f32
        }
        %alloc_29 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
        memref.copy %alloc_28, %alloc_29 : memref<768xf32> to memref<768xf32>
        linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_15, %alloc_12 : memref<768x768xf32, strided<[768, 1], offset: ?>>, memref<768xf32>) outs(%alloc_29 : memref<768xf32>) {
        ^bb0(%in: f32, %in_32: f32, %out: f32):
          %5 = arith.mulf %in, %in_32 : f32
          %6 = arith.addf %out, %5 : f32
          linalg.yield %6 : f32
        }
        linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_16, %alloc_12 : memref<768x768xf32, strided<[768, 1], offset: ?>>, memref<768xf32>) outs(%alloc_28 : memref<768xf32>) {
        ^bb0(%in: f32, %in_32: f32, %out: f32):
          %5 = arith.mulf %in, %in_32 : f32
          %6 = arith.addf %out, %5 : f32
          linalg.yield %6 : f32
        }
        %subview_30 = memref.subview %arg2[%arg17, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
        memref.copy %alloc_29, %subview_30 : memref<768xf32> to memref<768xf32, strided<[1], offset: ?>>
        %subview_31 = memref.subview %arg3[%arg17, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
        memref.copy %alloc_28, %subview_31 : memref<768xf32> to memref<768xf32, strided<[1], offset: ?>>
        cinm.yield
      }
      %2 = arith.index_cast %arg1 : index to i64
      %3 = arith.uitofp %2 : i64 to f32
      %subview_17 = memref.subview %arg2[%arg17, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
      scf.for %arg19 = %c0 to %c768 step %c2 {
        %5 = arith.remui %arg19, %c48 : index
        %6 = arith.index_cast %5 : index to i64
        %7 = arith.uitofp %6 : i64 to f32
        %8 = arith.divf %7, %cst_0 : f32
        %9 = math.powf %cst, %8 : f32
        %10 = arith.divf %cst_1, %9 : f32
        %11 = arith.mulf %3, %10 : f32
        %12 = math.cos %11 : f32
        %13 = math.sin %11 : f32
        %14 = arith.addi %arg19, %c1 : index
        %15 = memref.load %alloc_11[%arg19] : memref<768xf32>
        %16 = memref.load %alloc_11[%14] : memref<768xf32>
        %17 = arith.mulf %15, %12 : f32
        %18 = arith.mulf %16, %13 : f32
        %19 = arith.subf %17, %18 : f32
        memref.store %19, %alloc_11[%arg19] : memref<768xf32>
        %20 = arith.mulf %15, %13 : f32
        %21 = arith.mulf %16, %12 : f32
        %22 = arith.addf %20, %21 : f32
        memref.store %22, %alloc_11[%14] : memref<768xf32>
        %23 = arith.cmpi ult, %arg19, %c768 : index
        scf.if %23 {
          %24 = memref.load %subview_17[%arg19] : memref<768xf32, strided<[1], offset: ?>>
          %25 = memref.load %subview_17[%14] : memref<768xf32, strided<[1], offset: ?>>
          %26 = arith.mulf %24, %12 : f32
          %27 = arith.mulf %25, %13 : f32
          %28 = arith.subf %26, %27 : f32
          memref.store %28, %subview_17[%arg19] : memref<768xf32, strided<[1], offset: ?>>
          %29 = arith.mulf %24, %13 : f32
          %30 = arith.mulf %25, %12 : f32
          %31 = arith.addf %29, %30 : f32
          memref.store %31, %subview_17[%14] : memref<768xf32, strided<[1], offset: ?>>
        }
      }
      %subview_18 = memref.subview %arg2[%arg17, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: ?>>
      %subview_19 = memref.subview %arg3[%arg17, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: ?>>
      %4 = arith.addi %arg1, %c1 : index
      %alloc_20 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
      %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
      memref.copy %alloc_11, %alloc_21 : memref<768xf32> to memref<768xf32>
      scf.for %arg19 = %c0 to %c6 step %c1 {
        %5 = arith.muli %arg19, %c48 : index
        %alloc_28 = memref.alloc() {alignment = 64 : i64} : memref<1024xf32>
        scf.for %arg20 = %c0 to %4 step %c1 {
          %subview_30 = memref.subview %alloc_11[%5] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: ?>>
          %subview_31 = memref.subview %subview_18[%arg20, %5] [1, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: ?>> to memref<48xf32, strided<[1], offset: ?>>
          %6 = cinm.compute_ -> f32
               attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 1, 8>} {
            %alloc_32 = memref.alloc() {alignment = 64 : i64} : memref<f32>
            linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_32 : memref<f32>) {
            ^bb0(%out: f32):
              linalg.yield %cst_6 : f32
            }
            linalg.generic {indexing_maps = [#map1, #map1, #map2], iterator_types = ["reduction"]} ins(%subview_30, %subview_31 : memref<48xf32, strided<[1], offset: ?>>, memref<48xf32, strided<[1], offset: ?>>) outs(%alloc_32 : memref<f32>) {
            ^bb0(%in: f32, %in_33: f32, %out: f32):
              %9 = arith.mulf %in, %in_33 : f32
              %10 = arith.addf %9, %out : f32
              linalg.yield %10 : f32
            }
            %7 = memref.load %alloc_32[] : memref<f32>
            %8 = arith.divf %7, %cst_4 : f32
            cinm.yield %8 : f32
          }
          memref.store %6, %alloc_28[%arg20] : memref<1024xf32>
        }
        scf.for %arg20 = %4 to %c1024 step %c1 {
          memref.store %cst_5, %alloc_28[%arg20] : memref<1024xf32>
        }
        cinm.compute_
             attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
          %alloc_30 = memref.alloc() {alignment = 64 : i64} : memref<f32>
          linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_30 : memref<f32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_5 : f32
          }
          linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_28 : memref<1024xf32>) outs(%alloc_30 : memref<f32>) {
          ^bb0(%in: f32, %out: f32):
            %8 = arith.maxnumf %in, %out : f32
            linalg.yield %8 : f32
          }
          %6 = memref.load %alloc_30[] : memref<f32>
          linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%alloc_28 : memref<1024xf32>) outs(%alloc_28 : memref<1024xf32>) {
          ^bb0(%in: f32, %out: f32):
            %8 = arith.subf %in, %6 : f32
            %9 = math.exp %8 : f32
            linalg.yield %9 : f32
          }
          linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_30 : memref<f32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_6 : f32
          }
          linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_28 : memref<1024xf32>) outs(%alloc_30 : memref<f32>) {
          ^bb0(%in: f32, %out: f32):
            %8 = arith.addf %in, %out : f32
            linalg.yield %8 : f32
          }
          %7 = memref.load %alloc_30[] : memref<f32>
          linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%alloc_28 : memref<1024xf32>) outs(%alloc_28 : memref<1024xf32>) {
          ^bb0(%in: f32, %out: f32):
            %8 = arith.divf %in, %7 : f32
            linalg.yield %8 : f32
          }
          cinm.yield
        }
        %subview_29 = memref.subview %alloc_21[%5] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: ?>>
        linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%subview_29 : memref<48xf32, strided<[1], offset: ?>>) {
        ^bb0(%out: f32):
          linalg.yield %cst_6 : f32
        }
        scf.for %arg20 = %c0 to %4 step %c1 {
          %subview_30 = memref.subview %subview_19[%arg20, %5] [1, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: ?>> to memref<48xf32, strided<[1], offset: ?>>
          %6 = memref.load %alloc_28[%arg20] : memref<1024xf32>
          cinm.compute_
               attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
            linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%subview_29, %subview_30 : memref<48xf32, strided<[1], offset: ?>>, memref<48xf32, strided<[1], offset: ?>>) outs(%subview_29 : memref<48xf32, strided<[1], offset: ?>>) {
            ^bb0(%in: f32, %in_31: f32, %out: f32):
              %7 = arith.mulf %in_31, %6 : f32
              %8 = arith.addf %in, %7 : f32
              linalg.yield %8 : f32
            }
            cinm.yield
          }
        }
      }
      memref.copy %alloc_21, %alloc_11 : memref<768xf32> to memref<768xf32>
      %subview_22 = memref.subview %arg9[%arg17, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: ?>>
      cinm.compute_
           attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
        memref.copy %alloc_11, %alloc_10 : memref<768xf32> to memref<768xf32>
        linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_22, %alloc_11 : memref<768x768xf32, strided<[768, 1], offset: ?>>, memref<768xf32>) outs(%alloc_10 : memref<768xf32>) {
        ^bb0(%in: f32, %in_28: f32, %out: f32):
          %5 = arith.mulf %in, %in_28 : f32
          %6 = arith.addf %out, %5 : f32
          linalg.yield %6 : f32
        }
        linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%arg18, %alloc_10 : memref<768xf32, strided<[?], offset: ?>>, memref<768xf32>) outs(%alloc_10 : memref<768xf32>) {
        ^bb0(%in: f32, %in_28: f32, %out: f32):
          %5 = arith.addf %in, %in_28 : f32
          linalg.yield %5 : f32
        }
        cinm.yield
      }
      %subview_23 = memref.subview %arg13[%arg17, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: ?>>
      cinm.compute_
           attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
        %alloc_28 = memref.alloc() {alignment = 64 : i64} : memref<f32>
        linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_28 : memref<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_6 : f32
        }
        linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_10 : memref<768xf32>) outs(%alloc_28 : memref<f32>) {
        ^bb0(%in: f32, %out: f32):
          %9 = arith.mulf %in, %in : f32
          %10 = arith.addf %9, %out : f32
          linalg.yield %10 : f32
        }
        %5 = memref.load %alloc_28[] : memref<f32>
        %6 = arith.divf %5, %cst_3 : f32
        %7 = arith.addf %6, %cst_2 : f32
        %8 = math.rsqrt %7 : f32
        linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%alloc_10, %subview_23 : memref<768xf32>, memref<768xf32, strided<[1], offset: ?>>) outs(%alloc_20 : memref<768xf32>) {
        ^bb0(%in: f32, %in_29: f32, %out: f32):
          %9 = arith.mulf %in, %8 : f32
          %10 = arith.mulf %9, %in_29 : f32
          linalg.yield %10 : f32
        }
        cinm.yield
      }
      %subview_24 = memref.subview %arg10[%arg17, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: ?>>
      %subview_25 = memref.subview %arg12[%arg17, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: ?>>
      cinm.compute_
           attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
        linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%alloc_9 : memref<2048xf32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_6 : f32
        }
        memref.copy %alloc_9, %alloc_8 : memref<2048xf32> to memref<2048xf32>
        linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_24, %alloc_20 : memref<2048x768xf32, strided<[768, 1], offset: ?>>, memref<768xf32>) outs(%alloc_8 : memref<2048xf32>) {
        ^bb0(%in: f32, %in_28: f32, %out: f32):
          %5 = arith.mulf %in, %in_28 : f32
          %6 = arith.addf %out, %5 : f32
          linalg.yield %6 : f32
        }
        linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_25, %alloc_20 : memref<2048x768xf32, strided<[768, 1], offset: ?>>, memref<768xf32>) outs(%alloc_9 : memref<2048xf32>) {
        ^bb0(%in: f32, %in_28: f32, %out: f32):
          %5 = arith.mulf %in, %in_28 : f32
          %6 = arith.addf %out, %5 : f32
          linalg.yield %6 : f32
        }
        cinm.yield
      }
      linalg.map ins(%alloc_8, %alloc_9 : memref<2048xf32>, memref<2048xf32>) outs(%alloc_8 : memref<2048xf32>)
        (%in: f32, %in_28: f32) {
          %5 = arith.negf %in : f32
          %6 = math.exp %5 : f32
          %7 = arith.addf %6, %cst_1 : f32
          %8 = arith.divf %cst_1, %7 : f32
          %9 = arith.mulf %in, %8 : f32
          %10 = arith.mulf %9, %in_28 : f32
          linalg.yield %10 : f32
        }
      %subview_26 = memref.subview %arg11[%arg17, 0, 0] [1, 768, 2048] [1, 1, 1] : memref<6x768x2048xf32> to memref<768x2048xf32, strided<[2048, 1], offset: ?>>
      cinm.compute_
           attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
        linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_26, %alloc_8 : memref<768x2048xf32, strided<[2048, 1], offset: ?>>, memref<2048xf32>) outs(%alloc_20 : memref<768xf32>) {
        ^bb0(%in: f32, %in_28: f32, %out: f32):
          %5 = arith.mulf %in, %in_28 : f32
          %6 = arith.addf %out, %5 : f32
          linalg.yield %6 : f32
        }
        cinm.yield
      }
      %cast_27 = memref.cast %alloc_20 : memref<768xf32> to memref<768xf32, strided<[?], offset: ?>>
      scf.yield %cast_27 : memref<768xf32, strided<[?], offset: ?>>
    }
    cinm.compute_
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_8 : memref<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_6 : f32
      }
      linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%0 : memref<768xf32, strided<[?], offset: ?>>) outs(%alloc_8 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %6 = arith.mulf %in, %in : f32
        %7 = arith.addf %6, %out : f32
        linalg.yield %7 : f32
      }
      %2 = memref.load %alloc_8[] : memref<f32>
      %3 = arith.divf %2, %cst_3 : f32
      %4 = arith.addf %3, %cst_2 : f32
      %5 = math.rsqrt %4 : f32
      linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%0, %arg14 : memref<768xf32, strided<[?], offset: ?>>, memref<768xf32>) outs(%alloc_7 : memref<768xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %6 = arith.mulf %in, %5 : f32
        %7 = arith.mulf %6, %in_9 : f32
        linalg.yield %7 : f32
      }
      cinm.yield
    }
    %1 = cinm.compute_ -> memref<32000xf32, strided<[1]>>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 8, 16>} {
      %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<34048x768xf32>
      linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%alloc_8 : memref<34048x768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_6 : f32
      }
      %subview_9 = memref.subview %alloc_8[0, 0] [32000, 768] [1, 1] : memref<34048x768xf32> to memref<32000x768xf32, strided<[768, 1]>>
      memref.copy %arg15, %subview_9 : memref<32000x768xf32> to memref<32000x768xf32, strided<[768, 1]>>
      linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%alloc : memref<34048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_6 : f32
      }
      linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%alloc_8, %alloc_7 : memref<34048x768xf32>, memref<768xf32>) outs(%alloc : memref<34048xf32>) {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %2 = arith.mulf %in, %in_11 : f32
        %3 = arith.addf %out, %2 : f32
        linalg.yield %3 : f32
      }
      %subview_10 = memref.subview %alloc[0] [32000] [1] : memref<34048xf32> to memref<32000xf32, strided<[1]>>
      cinm.yield %subview_10 : memref<32000xf32, strided<[1]>>
    }
    memref.copy %1, %arg16 : memref<32000xf32, strided<[1]>> to memref<32000xf32, strided<[1]>>
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
        %2 = cinm.compute_ -> f32
             attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 1, 8>} {
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
      cinm.compute_
           attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
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
        cinm.compute_
             attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
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
    cinm.compute_
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
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
    cinm.compute_
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
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
}

