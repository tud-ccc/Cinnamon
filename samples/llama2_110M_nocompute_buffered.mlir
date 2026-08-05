#map = affine_map<() -> ()>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> ()>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d1)>
#map5 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @forward(%arg0: index, %arg1: index, %arg2: memref<6x1024x768xf32>, %arg3: memref<6x1024x768xf32>, %arg4: memref<32000x768xf32>, %arg5: memref<6x768xf32>, %arg6: memref<6x768x768xf32>, %arg7: memref<6x768x768xf32>, %arg8: memref<6x768x768xf32>, %arg9: memref<6x768x768xf32>, %arg10: memref<6x2048x768xf32>, %arg11: memref<6x768x2048xf32>, %arg12: memref<6x2048x768xf32>, %arg13: memref<6x768xf32>, %arg14: memref<768xf32>, %arg15: memref<32000x768xf32>, %arg16: memref<32000xf32, strided<[1]>>) {
    %cst = arith.constant 0xFFC00000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 0xFF800000 : f32
    %cst_2 = arith.constant 6.92820311 : f32
    %c1024 = arith.constant 1024 : index
    %c6 = arith.constant 6 : index
    %c1 = arith.constant 1 : index
    %cst_3 = arith.constant 7.680000e+02 : f32
    %cst_4 = arith.constant 9.99999974E-6 : f32
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c48 = arith.constant 48 : index
    %c768 = arith.constant 768 : index
    %cst_5 = arith.constant 1.000000e+00 : f32
    %cst_6 = arith.constant 4.800000e+01 : f32
    %cst_7 = arith.constant 1.000000e+04 : f32
    %subview = memref.subview %arg4[%arg0, 0] [1, 768] [1, 1] : memref<32000x768xf32> to memref<768xf32, strided<[1], offset: ?>>
    %subview_8 = memref.subview %arg5[0, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1]>>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<f32>
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_9 : memref<f32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_0 : f32
    }
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    %0 = memref.load %alloc_9[] : memref<f32>
    memref.store %0, %alloc_10[] : memref<f32>
    linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%subview : memref<768xf32, strided<[1], offset: ?>>) outs(%alloc_10 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %67 = arith.mulf %in, %in : f32
      %68 = arith.addf %67, %out : f32
      linalg.yield %68 : f32
    }
    %1 = memref.load %alloc_10[] : memref<f32>
    %2 = arith.divf %1, %cst_3 : f32
    %3 = arith.addf %2, %cst_4 : f32
    %4 = math.rsqrt %3 : f32
    %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%subview, %subview_8 : memref<768xf32, strided<[1], offset: ?>>, memref<768xf32, strided<[1]>>) outs(%alloc_12 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %4 : f32
      %68 = arith.mulf %67, %in_151 : f32
      linalg.yield %68 : f32
    }
    %subview_13 = memref.subview %arg6[0, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1]>>
    %subview_14 = memref.subview %arg7[0, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1]>>
    %subview_15 = memref.subview %arg8[0, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1]>>
    linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%alloc_11 : memref<768xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_0 : f32
    }
    %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc_11, %alloc_16 : memref<768xf32> to memref<768xf32>
    linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_13, %alloc_12 : memref<768x768xf32, strided<[768, 1]>>, memref<768xf32>) outs(%alloc_16 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %in_151 : f32
      %68 = arith.addf %out, %67 : f32
      linalg.yield %68 : f32
    }
    %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc_11, %alloc_17 : memref<768xf32> to memref<768xf32>
    linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_14, %alloc_12 : memref<768x768xf32, strided<[768, 1]>>, memref<768xf32>) outs(%alloc_17 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %in_151 : f32
      %68 = arith.addf %out, %67 : f32
      linalg.yield %68 : f32
    }
    %alloc_18 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc_11, %alloc_18 : memref<768xf32> to memref<768xf32>
    linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_15, %alloc_12 : memref<768x768xf32, strided<[768, 1]>>, memref<768xf32>) outs(%alloc_18 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %in_151 : f32
      %68 = arith.addf %out, %67 : f32
      linalg.yield %68 : f32
    }
    %subview_19 = memref.subview %arg3[0, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
    memref.copy %alloc_18, %subview_19 : memref<768xf32> to memref<768xf32, strided<[1], offset: ?>>
    %5 = arith.index_cast %arg1 : index to i64
    %6 = arith.uitofp %5 : i64 to f32
    scf.for %arg17 = %c0 to %c768 step %c2 {
      %67 = arith.remui %arg17, %c48 : index
      %68 = arith.index_cast %67 : index to i64
      %69 = arith.uitofp %68 : i64 to f32
      %70 = arith.divf %69, %cst_6 : f32
      %71 = math.powf %cst_7, %70 : f32
      %72 = arith.divf %cst_5, %71 : f32
      %73 = arith.mulf %6, %72 : f32
      %74 = math.cos %73 : f32
      %75 = math.sin %73 : f32
      %76 = arith.addi %arg17, %c1 : index
      %77 = memref.load %alloc_16[%arg17] : memref<768xf32>
      %78 = memref.load %alloc_16[%76] : memref<768xf32>
      %79 = arith.mulf %77, %74 : f32
      %80 = arith.mulf %78, %75 : f32
      %81 = arith.subf %79, %80 : f32
      memref.store %81, %alloc_16[%arg17] : memref<768xf32>
      %82 = arith.mulf %77, %75 : f32
      %83 = arith.mulf %78, %74 : f32
      %84 = arith.addf %82, %83 : f32
      memref.store %84, %alloc_16[%76] : memref<768xf32>
      %85 = arith.cmpi ult, %arg17, %c768 : index
      scf.if %85 {
        %86 = memref.load %alloc_17[%arg17] : memref<768xf32>
        %87 = memref.load %alloc_17[%76] : memref<768xf32>
        %88 = arith.mulf %86, %74 : f32
        %89 = arith.mulf %87, %75 : f32
        %90 = arith.subf %88, %89 : f32
        memref.store %90, %alloc_17[%arg17] : memref<768xf32>
        %91 = arith.mulf %86, %75 : f32
        %92 = arith.mulf %87, %74 : f32
        %93 = arith.addf %91, %92 : f32
        memref.store %93, %alloc_17[%76] : memref<768xf32>
      }
    }
    %subview_20 = memref.subview %arg2[0, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
    memref.copy %alloc_17, %subview_20 : memref<768xf32> to memref<768xf32, strided<[1], offset: ?>>
    %subview_21 = memref.subview %arg2[0, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1]>>
    %subview_22 = memref.subview %arg3[0, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1]>>
    %7 = arith.addi %arg1, %c1 : index
    %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc_16, %alloc_23 : memref<768xf32> to memref<768xf32>
    scf.for %arg17 = %c0 to %c6 step %c1 {
      %67 = arith.muli %arg17, %c48 : index
      %alloc_151 = memref.alloc() {alignment = 64 : i64} : memref<1024xf32>
      scf.for %arg18 = %c0 to %7 step %c1 {
        %subview_154 = memref.subview %alloc_16[%67] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: ?>>
        %subview_155 = memref.subview %subview_21[%arg18, %67] [1, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<48xf32, strided<[1], offset: ?>>
        %alloc_156 = memref.alloc() {alignment = 64 : i64} : memref<f32>
        %71 = memref.load %alloc_9[] : memref<f32>
        memref.store %71, %alloc_156[] : memref<f32>
        linalg.generic {indexing_maps = [#map1, #map1, #map2], iterator_types = ["reduction"]} ins(%subview_154, %subview_155 : memref<48xf32, strided<[1], offset: ?>>, memref<48xf32, strided<[1], offset: ?>>) outs(%alloc_156 : memref<f32>) {
        ^bb0(%in: f32, %in_157: f32, %out: f32):
          %74 = arith.mulf %in, %in_157 : f32
          %75 = arith.addf %74, %out : f32
          linalg.yield %75 : f32
        }
        %72 = memref.load %alloc_156[] : memref<f32>
        %73 = arith.divf %72, %cst_2 : f32
        memref.store %73, %alloc_151[%arg18] : memref<1024xf32>
      }
      scf.for %arg18 = %7 to %c1024 step %c1 {
        memref.store %cst_1, %alloc_151[%arg18] : memref<1024xf32>
      }
      linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc : memref<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      }
      linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_151 : memref<1024xf32>) outs(%alloc : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %71 = arith.maxnumf %in, %out : f32
        linalg.yield %71 : f32
      }
      %68 = memref.load %alloc[] : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%alloc_151 : memref<1024xf32>) outs(%alloc_151 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %71 = arith.subf %in, %68 : f32
        %72 = math.exp %71 : f32
        linalg.yield %72 : f32
      }
      %alloc_152 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      %69 = memref.load %alloc_9[] : memref<f32>
      memref.store %69, %alloc_152[] : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_151 : memref<1024xf32>) outs(%alloc_152 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %71 = arith.addf %in, %out : f32
        linalg.yield %71 : f32
      }
      %70 = memref.load %alloc_152[] : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%alloc_151 : memref<1024xf32>) outs(%alloc_151 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %71 = arith.divf %in, %70 : f32
        linalg.yield %71 : f32
      }
      %subview_153 = memref.subview %alloc_23[%67] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: ?>>
      linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%subview_153 : memref<48xf32, strided<[1], offset: ?>>) {
      ^bb0(%out: f32):
        linalg.yield %cst_0 : f32
      }
      scf.for %arg18 = %c0 to %7 step %c1 {
        %subview_154 = memref.subview %subview_22[%arg18, %67] [1, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<48xf32, strided<[1], offset: ?>>
        %71 = memref.load %alloc_151[%arg18] : memref<1024xf32>
        linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%subview_153, %subview_154 : memref<48xf32, strided<[1], offset: ?>>, memref<48xf32, strided<[1], offset: ?>>) outs(%subview_153 : memref<48xf32, strided<[1], offset: ?>>) {
        ^bb0(%in: f32, %in_155: f32, %out: f32):
          %72 = arith.mulf %in_155, %71 : f32
          %73 = arith.addf %in, %72 : f32
          linalg.yield %73 : f32
        }
      }
    }
    memref.copy %alloc_23, %alloc_16 : memref<768xf32> to memref<768xf32>
    %subview_24 = memref.subview %arg9[0, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1]>>
    %alloc_25 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc_16, %alloc_25 : memref<768xf32> to memref<768xf32>
    linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_24, %alloc_16 : memref<768x768xf32, strided<[768, 1]>>, memref<768xf32>) outs(%alloc_25 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %in_151 : f32
      %68 = arith.addf %out, %67 : f32
      linalg.yield %68 : f32
    }
    linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%subview, %alloc_25 : memref<768xf32, strided<[1], offset: ?>>, memref<768xf32>) outs(%alloc_25 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.addf %in, %in_151 : f32
      linalg.yield %67 : f32
    }
    %subview_26 = memref.subview %arg13[0, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1]>>
    %alloc_27 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    %8 = memref.load %alloc_9[] : memref<f32>
    memref.store %8, %alloc_27[] : memref<f32>
    linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_25 : memref<768xf32>) outs(%alloc_27 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %67 = arith.mulf %in, %in : f32
      %68 = arith.addf %67, %out : f32
      linalg.yield %68 : f32
    }
    %9 = memref.load %alloc_27[] : memref<f32>
    %10 = arith.divf %9, %cst_3 : f32
    %11 = arith.addf %10, %cst_4 : f32
    %12 = math.rsqrt %11 : f32
    %alloc_28 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%alloc_25, %subview_26 : memref<768xf32>, memref<768xf32, strided<[1]>>) outs(%alloc_28 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %12 : f32
      %68 = arith.mulf %67, %in_151 : f32
      linalg.yield %68 : f32
    }
    %subview_29 = memref.subview %arg10[0, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1]>>
    %subview_30 = memref.subview %arg12[0, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1]>>
    %alloc_31 = memref.alloc() {alignment = 64 : i64} : memref<2048xf32>
    linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%alloc_31 : memref<2048xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_0 : f32
    }
    %alloc_32 = memref.alloc() {alignment = 64 : i64} : memref<2048xf32>
    memref.copy %alloc_31, %alloc_32 : memref<2048xf32> to memref<2048xf32>
    linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_29, %alloc_28 : memref<2048x768xf32, strided<[768, 1]>>, memref<768xf32>) outs(%alloc_32 : memref<2048xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %in_151 : f32
      %68 = arith.addf %out, %67 : f32
      linalg.yield %68 : f32
    }
    %alloc_33 = memref.alloc() {alignment = 64 : i64} : memref<2048xf32>
    memref.copy %alloc_31, %alloc_33 : memref<2048xf32> to memref<2048xf32>
    linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_30, %alloc_28 : memref<2048x768xf32, strided<[768, 1]>>, memref<768xf32>) outs(%alloc_33 : memref<2048xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %in_151 : f32
      %68 = arith.addf %out, %67 : f32
      linalg.yield %68 : f32
    }
    linalg.map ins(%alloc_32, %alloc_33 : memref<2048xf32>, memref<2048xf32>) outs(%alloc_32 : memref<2048xf32>)
      (%in: f32, %in_151: f32) {
        %67 = arith.negf %in : f32
        %68 = math.exp %67 : f32
        %69 = arith.addf %68, %cst_5 : f32
        %70 = arith.divf %cst_5, %69 : f32
        %71 = arith.mulf %in, %70 : f32
        %72 = arith.mulf %71, %in_151 : f32
        linalg.yield %72 : f32
      }
    %subview_34 = memref.subview %arg11[0, 0, 0] [1, 768, 2048] [1, 1, 1] : memref<6x768x2048xf32> to memref<768x2048xf32, strided<[2048, 1]>>
    linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_34, %alloc_32 : memref<768x2048xf32, strided<[2048, 1]>>, memref<2048xf32>) outs(%alloc_28 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %in_151 : f32
      %68 = arith.addf %out, %67 : f32
      linalg.yield %68 : f32
    }
    %subview_35 = memref.subview %arg5[1, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: 768>>
    %alloc_36 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    %13 = memref.load %alloc_9[] : memref<f32>
    memref.store %13, %alloc_36[] : memref<f32>
    linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_28 : memref<768xf32>) outs(%alloc_36 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %67 = arith.mulf %in, %in : f32
      %68 = arith.addf %67, %out : f32
      linalg.yield %68 : f32
    }
    %14 = memref.load %alloc_36[] : memref<f32>
    %15 = arith.divf %14, %cst_3 : f32
    %16 = arith.addf %15, %cst_4 : f32
    %17 = math.rsqrt %16 : f32
    linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%alloc_28, %subview_35 : memref<768xf32>, memref<768xf32, strided<[1], offset: 768>>) outs(%alloc_12 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %17 : f32
      %68 = arith.mulf %67, %in_151 : f32
      linalg.yield %68 : f32
    }
    %subview_37 = memref.subview %arg6[1, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 589824>>
    %subview_38 = memref.subview %arg7[1, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 589824>>
    %subview_39 = memref.subview %arg8[1, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 589824>>
    %alloc_40 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc_11, %alloc_40 : memref<768xf32> to memref<768xf32>
    linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_37, %alloc_12 : memref<768x768xf32, strided<[768, 1], offset: 589824>>, memref<768xf32>) outs(%alloc_40 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %in_151 : f32
      %68 = arith.addf %out, %67 : f32
      linalg.yield %68 : f32
    }
    %alloc_41 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc_11, %alloc_41 : memref<768xf32> to memref<768xf32>
    linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_38, %alloc_12 : memref<768x768xf32, strided<[768, 1], offset: 589824>>, memref<768xf32>) outs(%alloc_41 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %in_151 : f32
      %68 = arith.addf %out, %67 : f32
      linalg.yield %68 : f32
    }
    %alloc_42 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc_11, %alloc_42 : memref<768xf32> to memref<768xf32>
    linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_39, %alloc_12 : memref<768x768xf32, strided<[768, 1], offset: 589824>>, memref<768xf32>) outs(%alloc_42 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %in_151 : f32
      %68 = arith.addf %out, %67 : f32
      linalg.yield %68 : f32
    }
    %subview_43 = memref.subview %arg3[1, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
    memref.copy %alloc_42, %subview_43 : memref<768xf32> to memref<768xf32, strided<[1], offset: ?>>
    scf.for %arg17 = %c0 to %c768 step %c2 {
      %67 = arith.remui %arg17, %c48 : index
      %68 = arith.index_cast %67 : index to i64
      %69 = arith.uitofp %68 : i64 to f32
      %70 = arith.divf %69, %cst_6 : f32
      %71 = math.powf %cst_7, %70 : f32
      %72 = arith.divf %cst_5, %71 : f32
      %73 = arith.mulf %6, %72 : f32
      %74 = math.cos %73 : f32
      %75 = math.sin %73 : f32
      %76 = arith.addi %arg17, %c1 : index
      %77 = memref.load %alloc_40[%arg17] : memref<768xf32>
      %78 = memref.load %alloc_40[%76] : memref<768xf32>
      %79 = arith.mulf %77, %74 : f32
      %80 = arith.mulf %78, %75 : f32
      %81 = arith.subf %79, %80 : f32
      memref.store %81, %alloc_40[%arg17] : memref<768xf32>
      %82 = arith.mulf %77, %75 : f32
      %83 = arith.mulf %78, %74 : f32
      %84 = arith.addf %82, %83 : f32
      memref.store %84, %alloc_40[%76] : memref<768xf32>
      %85 = arith.cmpi ult, %arg17, %c768 : index
      scf.if %85 {
        %86 = memref.load %alloc_41[%arg17] : memref<768xf32>
        %87 = memref.load %alloc_41[%76] : memref<768xf32>
        %88 = arith.mulf %86, %74 : f32
        %89 = arith.mulf %87, %75 : f32
        %90 = arith.subf %88, %89 : f32
        memref.store %90, %alloc_41[%arg17] : memref<768xf32>
        %91 = arith.mulf %86, %75 : f32
        %92 = arith.mulf %87, %74 : f32
        %93 = arith.addf %91, %92 : f32
        memref.store %93, %alloc_41[%76] : memref<768xf32>
      }
    }
    %subview_44 = memref.subview %arg2[1, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
    memref.copy %alloc_41, %subview_44 : memref<768xf32> to memref<768xf32, strided<[1], offset: ?>>
    %subview_45 = memref.subview %arg2[1, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 786432>>
    %subview_46 = memref.subview %arg3[1, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 786432>>
    %alloc_47 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc_40, %alloc_47 : memref<768xf32> to memref<768xf32>
    scf.for %arg17 = %c0 to %c6 step %c1 {
      %67 = arith.muli %arg17, %c48 : index
      %alloc_151 = memref.alloc() {alignment = 64 : i64} : memref<1024xf32>
      scf.for %arg18 = %c0 to %7 step %c1 {
        %subview_154 = memref.subview %alloc_40[%67] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: ?>>
        %subview_155 = memref.subview %subview_45[%arg18, %67] [1, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<48xf32, strided<[1], offset: ?>>
        %alloc_156 = memref.alloc() {alignment = 64 : i64} : memref<f32>
        %71 = memref.load %alloc_9[] : memref<f32>
        memref.store %71, %alloc_156[] : memref<f32>
        linalg.generic {indexing_maps = [#map1, #map1, #map2], iterator_types = ["reduction"]} ins(%subview_154, %subview_155 : memref<48xf32, strided<[1], offset: ?>>, memref<48xf32, strided<[1], offset: ?>>) outs(%alloc_156 : memref<f32>) {
        ^bb0(%in: f32, %in_157: f32, %out: f32):
          %74 = arith.mulf %in, %in_157 : f32
          %75 = arith.addf %74, %out : f32
          linalg.yield %75 : f32
        }
        %72 = memref.load %alloc_156[] : memref<f32>
        %73 = arith.divf %72, %cst_2 : f32
        memref.store %73, %alloc_151[%arg18] : memref<1024xf32>
      }
      scf.for %arg18 = %7 to %c1024 step %c1 {
        memref.store %cst_1, %alloc_151[%arg18] : memref<1024xf32>
      }
      linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc : memref<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      }
      linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_151 : memref<1024xf32>) outs(%alloc : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %71 = arith.maxnumf %in, %out : f32
        linalg.yield %71 : f32
      }
      %68 = memref.load %alloc[] : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%alloc_151 : memref<1024xf32>) outs(%alloc_151 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %71 = arith.subf %in, %68 : f32
        %72 = math.exp %71 : f32
        linalg.yield %72 : f32
      }
      %alloc_152 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      %69 = memref.load %alloc_9[] : memref<f32>
      memref.store %69, %alloc_152[] : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_151 : memref<1024xf32>) outs(%alloc_152 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %71 = arith.addf %in, %out : f32
        linalg.yield %71 : f32
      }
      %70 = memref.load %alloc_152[] : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%alloc_151 : memref<1024xf32>) outs(%alloc_151 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %71 = arith.divf %in, %70 : f32
        linalg.yield %71 : f32
      }
      %subview_153 = memref.subview %alloc_47[%67] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: ?>>
      linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%subview_153 : memref<48xf32, strided<[1], offset: ?>>) {
      ^bb0(%out: f32):
        linalg.yield %cst_0 : f32
      }
      scf.for %arg18 = %c0 to %7 step %c1 {
        %subview_154 = memref.subview %subview_46[%arg18, %67] [1, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<48xf32, strided<[1], offset: ?>>
        %71 = memref.load %alloc_151[%arg18] : memref<1024xf32>
        linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%subview_153, %subview_154 : memref<48xf32, strided<[1], offset: ?>>, memref<48xf32, strided<[1], offset: ?>>) outs(%subview_153 : memref<48xf32, strided<[1], offset: ?>>) {
        ^bb0(%in: f32, %in_155: f32, %out: f32):
          %72 = arith.mulf %in_155, %71 : f32
          %73 = arith.addf %in, %72 : f32
          linalg.yield %73 : f32
        }
      }
    }
    memref.copy %alloc_47, %alloc_40 : memref<768xf32> to memref<768xf32>
    %subview_48 = memref.subview %arg9[1, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 589824>>
    %alloc_49 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc_40, %alloc_49 : memref<768xf32> to memref<768xf32>
    linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_48, %alloc_40 : memref<768x768xf32, strided<[768, 1], offset: 589824>>, memref<768xf32>) outs(%alloc_49 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %in_151 : f32
      %68 = arith.addf %out, %67 : f32
      linalg.yield %68 : f32
    }
    linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%alloc_28, %alloc_49 : memref<768xf32>, memref<768xf32>) outs(%alloc_49 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.addf %in, %in_151 : f32
      linalg.yield %67 : f32
    }
    %subview_50 = memref.subview %arg13[1, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: 768>>
    %alloc_51 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    %18 = memref.load %alloc_9[] : memref<f32>
    memref.store %18, %alloc_51[] : memref<f32>
    linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_49 : memref<768xf32>) outs(%alloc_51 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %67 = arith.mulf %in, %in : f32
      %68 = arith.addf %67, %out : f32
      linalg.yield %68 : f32
    }
    %19 = memref.load %alloc_51[] : memref<f32>
    %20 = arith.divf %19, %cst_3 : f32
    %21 = arith.addf %20, %cst_4 : f32
    %22 = math.rsqrt %21 : f32
    %alloc_52 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%alloc_49, %subview_50 : memref<768xf32>, memref<768xf32, strided<[1], offset: 768>>) outs(%alloc_52 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %22 : f32
      %68 = arith.mulf %67, %in_151 : f32
      linalg.yield %68 : f32
    }
    %subview_53 = memref.subview %arg10[1, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: 1572864>>
    %subview_54 = memref.subview %arg12[1, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: 1572864>>
    %alloc_55 = memref.alloc() {alignment = 64 : i64} : memref<2048xf32>
    memref.copy %alloc_31, %alloc_55 : memref<2048xf32> to memref<2048xf32>
    linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_53, %alloc_52 : memref<2048x768xf32, strided<[768, 1], offset: 1572864>>, memref<768xf32>) outs(%alloc_55 : memref<2048xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %in_151 : f32
      %68 = arith.addf %out, %67 : f32
      linalg.yield %68 : f32
    }
    %alloc_56 = memref.alloc() {alignment = 64 : i64} : memref<2048xf32>
    memref.copy %alloc_31, %alloc_56 : memref<2048xf32> to memref<2048xf32>
    linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_54, %alloc_52 : memref<2048x768xf32, strided<[768, 1], offset: 1572864>>, memref<768xf32>) outs(%alloc_56 : memref<2048xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %in_151 : f32
      %68 = arith.addf %out, %67 : f32
      linalg.yield %68 : f32
    }
    linalg.map ins(%alloc_55, %alloc_56 : memref<2048xf32>, memref<2048xf32>) outs(%alloc_55 : memref<2048xf32>)
      (%in: f32, %in_151: f32) {
        %67 = arith.negf %in : f32
        %68 = math.exp %67 : f32
        %69 = arith.addf %68, %cst_5 : f32
        %70 = arith.divf %cst_5, %69 : f32
        %71 = arith.mulf %in, %70 : f32
        %72 = arith.mulf %71, %in_151 : f32
        linalg.yield %72 : f32
      }
    %subview_57 = memref.subview %arg11[1, 0, 0] [1, 768, 2048] [1, 1, 1] : memref<6x768x2048xf32> to memref<768x2048xf32, strided<[2048, 1], offset: 1572864>>
    linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_57, %alloc_55 : memref<768x2048xf32, strided<[2048, 1], offset: 1572864>>, memref<2048xf32>) outs(%alloc_52 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %in_151 : f32
      %68 = arith.addf %out, %67 : f32
      linalg.yield %68 : f32
    }
    %subview_58 = memref.subview %arg5[2, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: 1536>>
    %alloc_59 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    %23 = memref.load %alloc_9[] : memref<f32>
    memref.store %23, %alloc_59[] : memref<f32>
    linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_52 : memref<768xf32>) outs(%alloc_59 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %67 = arith.mulf %in, %in : f32
      %68 = arith.addf %67, %out : f32
      linalg.yield %68 : f32
    }
    %24 = memref.load %alloc_59[] : memref<f32>
    %25 = arith.divf %24, %cst_3 : f32
    %26 = arith.addf %25, %cst_4 : f32
    %27 = math.rsqrt %26 : f32
    linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%alloc_52, %subview_58 : memref<768xf32>, memref<768xf32, strided<[1], offset: 1536>>) outs(%alloc_12 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %27 : f32
      %68 = arith.mulf %67, %in_151 : f32
      linalg.yield %68 : f32
    }
    %subview_60 = memref.subview %arg6[2, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 1179648>>
    %subview_61 = memref.subview %arg7[2, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 1179648>>
    %subview_62 = memref.subview %arg8[2, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 1179648>>
    %alloc_63 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc_11, %alloc_63 : memref<768xf32> to memref<768xf32>
    linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_60, %alloc_12 : memref<768x768xf32, strided<[768, 1], offset: 1179648>>, memref<768xf32>) outs(%alloc_63 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %in_151 : f32
      %68 = arith.addf %out, %67 : f32
      linalg.yield %68 : f32
    }
    %alloc_64 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc_11, %alloc_64 : memref<768xf32> to memref<768xf32>
    linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_61, %alloc_12 : memref<768x768xf32, strided<[768, 1], offset: 1179648>>, memref<768xf32>) outs(%alloc_64 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %in_151 : f32
      %68 = arith.addf %out, %67 : f32
      linalg.yield %68 : f32
    }
    %alloc_65 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc_11, %alloc_65 : memref<768xf32> to memref<768xf32>
    linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_62, %alloc_12 : memref<768x768xf32, strided<[768, 1], offset: 1179648>>, memref<768xf32>) outs(%alloc_65 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %in_151 : f32
      %68 = arith.addf %out, %67 : f32
      linalg.yield %68 : f32
    }
    %subview_66 = memref.subview %arg3[2, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
    memref.copy %alloc_65, %subview_66 : memref<768xf32> to memref<768xf32, strided<[1], offset: ?>>
    scf.for %arg17 = %c0 to %c768 step %c2 {
      %67 = arith.remui %arg17, %c48 : index
      %68 = arith.index_cast %67 : index to i64
      %69 = arith.uitofp %68 : i64 to f32
      %70 = arith.divf %69, %cst_6 : f32
      %71 = math.powf %cst_7, %70 : f32
      %72 = arith.divf %cst_5, %71 : f32
      %73 = arith.mulf %6, %72 : f32
      %74 = math.cos %73 : f32
      %75 = math.sin %73 : f32
      %76 = arith.addi %arg17, %c1 : index
      %77 = memref.load %alloc_63[%arg17] : memref<768xf32>
      %78 = memref.load %alloc_63[%76] : memref<768xf32>
      %79 = arith.mulf %77, %74 : f32
      %80 = arith.mulf %78, %75 : f32
      %81 = arith.subf %79, %80 : f32
      memref.store %81, %alloc_63[%arg17] : memref<768xf32>
      %82 = arith.mulf %77, %75 : f32
      %83 = arith.mulf %78, %74 : f32
      %84 = arith.addf %82, %83 : f32
      memref.store %84, %alloc_63[%76] : memref<768xf32>
      %85 = arith.cmpi ult, %arg17, %c768 : index
      scf.if %85 {
        %86 = memref.load %alloc_64[%arg17] : memref<768xf32>
        %87 = memref.load %alloc_64[%76] : memref<768xf32>
        %88 = arith.mulf %86, %74 : f32
        %89 = arith.mulf %87, %75 : f32
        %90 = arith.subf %88, %89 : f32
        memref.store %90, %alloc_64[%arg17] : memref<768xf32>
        %91 = arith.mulf %86, %75 : f32
        %92 = arith.mulf %87, %74 : f32
        %93 = arith.addf %91, %92 : f32
        memref.store %93, %alloc_64[%76] : memref<768xf32>
      }
    }
    %subview_67 = memref.subview %arg2[2, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
    memref.copy %alloc_64, %subview_67 : memref<768xf32> to memref<768xf32, strided<[1], offset: ?>>
    %subview_68 = memref.subview %arg2[2, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 1572864>>
    %subview_69 = memref.subview %arg3[2, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 1572864>>
    %alloc_70 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc_63, %alloc_70 : memref<768xf32> to memref<768xf32>
    scf.for %arg17 = %c0 to %c6 step %c1 {
      %67 = arith.muli %arg17, %c48 : index
      %alloc_151 = memref.alloc() {alignment = 64 : i64} : memref<1024xf32>
      scf.for %arg18 = %c0 to %7 step %c1 {
        %subview_154 = memref.subview %alloc_63[%67] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: ?>>
        %subview_155 = memref.subview %subview_68[%arg18, %67] [1, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<48xf32, strided<[1], offset: ?>>
        %alloc_156 = memref.alloc() {alignment = 64 : i64} : memref<f32>
        %71 = memref.load %alloc_9[] : memref<f32>
        memref.store %71, %alloc_156[] : memref<f32>
        linalg.generic {indexing_maps = [#map1, #map1, #map2], iterator_types = ["reduction"]} ins(%subview_154, %subview_155 : memref<48xf32, strided<[1], offset: ?>>, memref<48xf32, strided<[1], offset: ?>>) outs(%alloc_156 : memref<f32>) {
        ^bb0(%in: f32, %in_157: f32, %out: f32):
          %74 = arith.mulf %in, %in_157 : f32
          %75 = arith.addf %74, %out : f32
          linalg.yield %75 : f32
        }
        %72 = memref.load %alloc_156[] : memref<f32>
        %73 = arith.divf %72, %cst_2 : f32
        memref.store %73, %alloc_151[%arg18] : memref<1024xf32>
      }
      scf.for %arg18 = %7 to %c1024 step %c1 {
        memref.store %cst_1, %alloc_151[%arg18] : memref<1024xf32>
      }
      linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc : memref<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      }
      linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_151 : memref<1024xf32>) outs(%alloc : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %71 = arith.maxnumf %in, %out : f32
        linalg.yield %71 : f32
      }
      %68 = memref.load %alloc[] : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%alloc_151 : memref<1024xf32>) outs(%alloc_151 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %71 = arith.subf %in, %68 : f32
        %72 = math.exp %71 : f32
        linalg.yield %72 : f32
      }
      %alloc_152 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      %69 = memref.load %alloc_9[] : memref<f32>
      memref.store %69, %alloc_152[] : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_151 : memref<1024xf32>) outs(%alloc_152 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %71 = arith.addf %in, %out : f32
        linalg.yield %71 : f32
      }
      %70 = memref.load %alloc_152[] : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%alloc_151 : memref<1024xf32>) outs(%alloc_151 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %71 = arith.divf %in, %70 : f32
        linalg.yield %71 : f32
      }
      %subview_153 = memref.subview %alloc_70[%67] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: ?>>
      linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%subview_153 : memref<48xf32, strided<[1], offset: ?>>) {
      ^bb0(%out: f32):
        linalg.yield %cst_0 : f32
      }
      scf.for %arg18 = %c0 to %7 step %c1 {
        %subview_154 = memref.subview %subview_69[%arg18, %67] [1, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<48xf32, strided<[1], offset: ?>>
        %71 = memref.load %alloc_151[%arg18] : memref<1024xf32>
        linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%subview_153, %subview_154 : memref<48xf32, strided<[1], offset: ?>>, memref<48xf32, strided<[1], offset: ?>>) outs(%subview_153 : memref<48xf32, strided<[1], offset: ?>>) {
        ^bb0(%in: f32, %in_155: f32, %out: f32):
          %72 = arith.mulf %in_155, %71 : f32
          %73 = arith.addf %in, %72 : f32
          linalg.yield %73 : f32
        }
      }
    }
    memref.copy %alloc_70, %alloc_63 : memref<768xf32> to memref<768xf32>
    %subview_71 = memref.subview %arg9[2, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 1179648>>
    %alloc_72 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc_63, %alloc_72 : memref<768xf32> to memref<768xf32>
    linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_71, %alloc_63 : memref<768x768xf32, strided<[768, 1], offset: 1179648>>, memref<768xf32>) outs(%alloc_72 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %in_151 : f32
      %68 = arith.addf %out, %67 : f32
      linalg.yield %68 : f32
    }
    linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%alloc_52, %alloc_72 : memref<768xf32>, memref<768xf32>) outs(%alloc_72 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.addf %in, %in_151 : f32
      linalg.yield %67 : f32
    }
    %subview_73 = memref.subview %arg13[2, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: 1536>>
    %alloc_74 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    %28 = memref.load %alloc_9[] : memref<f32>
    memref.store %28, %alloc_74[] : memref<f32>
    linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_72 : memref<768xf32>) outs(%alloc_74 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %67 = arith.mulf %in, %in : f32
      %68 = arith.addf %67, %out : f32
      linalg.yield %68 : f32
    }
    %29 = memref.load %alloc_74[] : memref<f32>
    %30 = arith.divf %29, %cst_3 : f32
    %31 = arith.addf %30, %cst_4 : f32
    %32 = math.rsqrt %31 : f32
    %alloc_75 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%alloc_72, %subview_73 : memref<768xf32>, memref<768xf32, strided<[1], offset: 1536>>) outs(%alloc_75 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %32 : f32
      %68 = arith.mulf %67, %in_151 : f32
      linalg.yield %68 : f32
    }
    %subview_76 = memref.subview %arg10[2, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: 3145728>>
    %subview_77 = memref.subview %arg12[2, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: 3145728>>
    %alloc_78 = memref.alloc() {alignment = 64 : i64} : memref<2048xf32>
    memref.copy %alloc_31, %alloc_78 : memref<2048xf32> to memref<2048xf32>
    linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_76, %alloc_75 : memref<2048x768xf32, strided<[768, 1], offset: 3145728>>, memref<768xf32>) outs(%alloc_78 : memref<2048xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %in_151 : f32
      %68 = arith.addf %out, %67 : f32
      linalg.yield %68 : f32
    }
    %alloc_79 = memref.alloc() {alignment = 64 : i64} : memref<2048xf32>
    memref.copy %alloc_31, %alloc_79 : memref<2048xf32> to memref<2048xf32>
    linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_77, %alloc_75 : memref<2048x768xf32, strided<[768, 1], offset: 3145728>>, memref<768xf32>) outs(%alloc_79 : memref<2048xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %in_151 : f32
      %68 = arith.addf %out, %67 : f32
      linalg.yield %68 : f32
    }
    linalg.map ins(%alloc_78, %alloc_79 : memref<2048xf32>, memref<2048xf32>) outs(%alloc_78 : memref<2048xf32>)
      (%in: f32, %in_151: f32) {
        %67 = arith.negf %in : f32
        %68 = math.exp %67 : f32
        %69 = arith.addf %68, %cst_5 : f32
        %70 = arith.divf %cst_5, %69 : f32
        %71 = arith.mulf %in, %70 : f32
        %72 = arith.mulf %71, %in_151 : f32
        linalg.yield %72 : f32
      }
    %subview_80 = memref.subview %arg11[2, 0, 0] [1, 768, 2048] [1, 1, 1] : memref<6x768x2048xf32> to memref<768x2048xf32, strided<[2048, 1], offset: 3145728>>
    linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_80, %alloc_78 : memref<768x2048xf32, strided<[2048, 1], offset: 3145728>>, memref<2048xf32>) outs(%alloc_75 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %in_151 : f32
      %68 = arith.addf %out, %67 : f32
      linalg.yield %68 : f32
    }
    %subview_81 = memref.subview %arg5[3, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: 2304>>
    %alloc_82 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    %33 = memref.load %alloc_9[] : memref<f32>
    memref.store %33, %alloc_82[] : memref<f32>
    linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_75 : memref<768xf32>) outs(%alloc_82 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %67 = arith.mulf %in, %in : f32
      %68 = arith.addf %67, %out : f32
      linalg.yield %68 : f32
    }
    %34 = memref.load %alloc_82[] : memref<f32>
    %35 = arith.divf %34, %cst_3 : f32
    %36 = arith.addf %35, %cst_4 : f32
    %37 = math.rsqrt %36 : f32
    linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%alloc_75, %subview_81 : memref<768xf32>, memref<768xf32, strided<[1], offset: 2304>>) outs(%alloc_12 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %37 : f32
      %68 = arith.mulf %67, %in_151 : f32
      linalg.yield %68 : f32
    }
    %subview_83 = memref.subview %arg6[3, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 1769472>>
    %subview_84 = memref.subview %arg7[3, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 1769472>>
    %subview_85 = memref.subview %arg8[3, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 1769472>>
    %alloc_86 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc_11, %alloc_86 : memref<768xf32> to memref<768xf32>
    linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_83, %alloc_12 : memref<768x768xf32, strided<[768, 1], offset: 1769472>>, memref<768xf32>) outs(%alloc_86 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %in_151 : f32
      %68 = arith.addf %out, %67 : f32
      linalg.yield %68 : f32
    }
    %alloc_87 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc_11, %alloc_87 : memref<768xf32> to memref<768xf32>
    linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_84, %alloc_12 : memref<768x768xf32, strided<[768, 1], offset: 1769472>>, memref<768xf32>) outs(%alloc_87 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %in_151 : f32
      %68 = arith.addf %out, %67 : f32
      linalg.yield %68 : f32
    }
    %alloc_88 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc_11, %alloc_88 : memref<768xf32> to memref<768xf32>
    linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_85, %alloc_12 : memref<768x768xf32, strided<[768, 1], offset: 1769472>>, memref<768xf32>) outs(%alloc_88 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %in_151 : f32
      %68 = arith.addf %out, %67 : f32
      linalg.yield %68 : f32
    }
    %subview_89 = memref.subview %arg3[3, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
    memref.copy %alloc_88, %subview_89 : memref<768xf32> to memref<768xf32, strided<[1], offset: ?>>
    scf.for %arg17 = %c0 to %c768 step %c2 {
      %67 = arith.remui %arg17, %c48 : index
      %68 = arith.index_cast %67 : index to i64
      %69 = arith.uitofp %68 : i64 to f32
      %70 = arith.divf %69, %cst_6 : f32
      %71 = math.powf %cst_7, %70 : f32
      %72 = arith.divf %cst_5, %71 : f32
      %73 = arith.mulf %6, %72 : f32
      %74 = math.cos %73 : f32
      %75 = math.sin %73 : f32
      %76 = arith.addi %arg17, %c1 : index
      %77 = memref.load %alloc_86[%arg17] : memref<768xf32>
      %78 = memref.load %alloc_86[%76] : memref<768xf32>
      %79 = arith.mulf %77, %74 : f32
      %80 = arith.mulf %78, %75 : f32
      %81 = arith.subf %79, %80 : f32
      memref.store %81, %alloc_86[%arg17] : memref<768xf32>
      %82 = arith.mulf %77, %75 : f32
      %83 = arith.mulf %78, %74 : f32
      %84 = arith.addf %82, %83 : f32
      memref.store %84, %alloc_86[%76] : memref<768xf32>
      %85 = arith.cmpi ult, %arg17, %c768 : index
      scf.if %85 {
        %86 = memref.load %alloc_87[%arg17] : memref<768xf32>
        %87 = memref.load %alloc_87[%76] : memref<768xf32>
        %88 = arith.mulf %86, %74 : f32
        %89 = arith.mulf %87, %75 : f32
        %90 = arith.subf %88, %89 : f32
        memref.store %90, %alloc_87[%arg17] : memref<768xf32>
        %91 = arith.mulf %86, %75 : f32
        %92 = arith.mulf %87, %74 : f32
        %93 = arith.addf %91, %92 : f32
        memref.store %93, %alloc_87[%76] : memref<768xf32>
      }
    }
    %subview_90 = memref.subview %arg2[3, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
    memref.copy %alloc_87, %subview_90 : memref<768xf32> to memref<768xf32, strided<[1], offset: ?>>
    %subview_91 = memref.subview %arg2[3, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 2359296>>
    %subview_92 = memref.subview %arg3[3, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 2359296>>
    %alloc_93 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc_86, %alloc_93 : memref<768xf32> to memref<768xf32>
    scf.for %arg17 = %c0 to %c6 step %c1 {
      %67 = arith.muli %arg17, %c48 : index
      %alloc_151 = memref.alloc() {alignment = 64 : i64} : memref<1024xf32>
      scf.for %arg18 = %c0 to %7 step %c1 {
        %subview_154 = memref.subview %alloc_86[%67] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: ?>>
        %subview_155 = memref.subview %subview_91[%arg18, %67] [1, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<48xf32, strided<[1], offset: ?>>
        %alloc_156 = memref.alloc() {alignment = 64 : i64} : memref<f32>
        %71 = memref.load %alloc_9[] : memref<f32>
        memref.store %71, %alloc_156[] : memref<f32>
        linalg.generic {indexing_maps = [#map1, #map1, #map2], iterator_types = ["reduction"]} ins(%subview_154, %subview_155 : memref<48xf32, strided<[1], offset: ?>>, memref<48xf32, strided<[1], offset: ?>>) outs(%alloc_156 : memref<f32>) {
        ^bb0(%in: f32, %in_157: f32, %out: f32):
          %74 = arith.mulf %in, %in_157 : f32
          %75 = arith.addf %74, %out : f32
          linalg.yield %75 : f32
        }
        %72 = memref.load %alloc_156[] : memref<f32>
        %73 = arith.divf %72, %cst_2 : f32
        memref.store %73, %alloc_151[%arg18] : memref<1024xf32>
      }
      scf.for %arg18 = %7 to %c1024 step %c1 {
        memref.store %cst_1, %alloc_151[%arg18] : memref<1024xf32>
      }
      linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc : memref<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      }
      linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_151 : memref<1024xf32>) outs(%alloc : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %71 = arith.maxnumf %in, %out : f32
        linalg.yield %71 : f32
      }
      %68 = memref.load %alloc[] : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%alloc_151 : memref<1024xf32>) outs(%alloc_151 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %71 = arith.subf %in, %68 : f32
        %72 = math.exp %71 : f32
        linalg.yield %72 : f32
      }
      %alloc_152 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      %69 = memref.load %alloc_9[] : memref<f32>
      memref.store %69, %alloc_152[] : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_151 : memref<1024xf32>) outs(%alloc_152 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %71 = arith.addf %in, %out : f32
        linalg.yield %71 : f32
      }
      %70 = memref.load %alloc_152[] : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%alloc_151 : memref<1024xf32>) outs(%alloc_151 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %71 = arith.divf %in, %70 : f32
        linalg.yield %71 : f32
      }
      %subview_153 = memref.subview %alloc_93[%67] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: ?>>
      linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%subview_153 : memref<48xf32, strided<[1], offset: ?>>) {
      ^bb0(%out: f32):
        linalg.yield %cst_0 : f32
      }
      scf.for %arg18 = %c0 to %7 step %c1 {
        %subview_154 = memref.subview %subview_92[%arg18, %67] [1, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<48xf32, strided<[1], offset: ?>>
        %71 = memref.load %alloc_151[%arg18] : memref<1024xf32>
        linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%subview_153, %subview_154 : memref<48xf32, strided<[1], offset: ?>>, memref<48xf32, strided<[1], offset: ?>>) outs(%subview_153 : memref<48xf32, strided<[1], offset: ?>>) {
        ^bb0(%in: f32, %in_155: f32, %out: f32):
          %72 = arith.mulf %in_155, %71 : f32
          %73 = arith.addf %in, %72 : f32
          linalg.yield %73 : f32
        }
      }
    }
    memref.copy %alloc_93, %alloc_86 : memref<768xf32> to memref<768xf32>
    %subview_94 = memref.subview %arg9[3, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 1769472>>
    %alloc_95 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc_86, %alloc_95 : memref<768xf32> to memref<768xf32>
    linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_94, %alloc_86 : memref<768x768xf32, strided<[768, 1], offset: 1769472>>, memref<768xf32>) outs(%alloc_95 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %in_151 : f32
      %68 = arith.addf %out, %67 : f32
      linalg.yield %68 : f32
    }
    linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%alloc_75, %alloc_95 : memref<768xf32>, memref<768xf32>) outs(%alloc_95 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.addf %in, %in_151 : f32
      linalg.yield %67 : f32
    }
    %subview_96 = memref.subview %arg13[3, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: 2304>>
    %alloc_97 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    %38 = memref.load %alloc_9[] : memref<f32>
    memref.store %38, %alloc_97[] : memref<f32>
    linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_95 : memref<768xf32>) outs(%alloc_97 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %67 = arith.mulf %in, %in : f32
      %68 = arith.addf %67, %out : f32
      linalg.yield %68 : f32
    }
    %39 = memref.load %alloc_97[] : memref<f32>
    %40 = arith.divf %39, %cst_3 : f32
    %41 = arith.addf %40, %cst_4 : f32
    %42 = math.rsqrt %41 : f32
    %alloc_98 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%alloc_95, %subview_96 : memref<768xf32>, memref<768xf32, strided<[1], offset: 2304>>) outs(%alloc_98 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %42 : f32
      %68 = arith.mulf %67, %in_151 : f32
      linalg.yield %68 : f32
    }
    %subview_99 = memref.subview %arg10[3, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: 4718592>>
    %subview_100 = memref.subview %arg12[3, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: 4718592>>
    %alloc_101 = memref.alloc() {alignment = 64 : i64} : memref<2048xf32>
    memref.copy %alloc_31, %alloc_101 : memref<2048xf32> to memref<2048xf32>
    linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_99, %alloc_98 : memref<2048x768xf32, strided<[768, 1], offset: 4718592>>, memref<768xf32>) outs(%alloc_101 : memref<2048xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %in_151 : f32
      %68 = arith.addf %out, %67 : f32
      linalg.yield %68 : f32
    }
    %alloc_102 = memref.alloc() {alignment = 64 : i64} : memref<2048xf32>
    memref.copy %alloc_31, %alloc_102 : memref<2048xf32> to memref<2048xf32>
    linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %alloc_98 : memref<2048x768xf32, strided<[768, 1], offset: 4718592>>, memref<768xf32>) outs(%alloc_102 : memref<2048xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %in_151 : f32
      %68 = arith.addf %out, %67 : f32
      linalg.yield %68 : f32
    }
    linalg.map ins(%alloc_101, %alloc_102 : memref<2048xf32>, memref<2048xf32>) outs(%alloc_101 : memref<2048xf32>)
      (%in: f32, %in_151: f32) {
        %67 = arith.negf %in : f32
        %68 = math.exp %67 : f32
        %69 = arith.addf %68, %cst_5 : f32
        %70 = arith.divf %cst_5, %69 : f32
        %71 = arith.mulf %in, %70 : f32
        %72 = arith.mulf %71, %in_151 : f32
        linalg.yield %72 : f32
      }
    %subview_103 = memref.subview %arg11[3, 0, 0] [1, 768, 2048] [1, 1, 1] : memref<6x768x2048xf32> to memref<768x2048xf32, strided<[2048, 1], offset: 4718592>>
    linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_103, %alloc_101 : memref<768x2048xf32, strided<[2048, 1], offset: 4718592>>, memref<2048xf32>) outs(%alloc_98 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %in_151 : f32
      %68 = arith.addf %out, %67 : f32
      linalg.yield %68 : f32
    }
    %subview_104 = memref.subview %arg5[4, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: 3072>>
    %alloc_105 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    %43 = memref.load %alloc_9[] : memref<f32>
    memref.store %43, %alloc_105[] : memref<f32>
    linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_98 : memref<768xf32>) outs(%alloc_105 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %67 = arith.mulf %in, %in : f32
      %68 = arith.addf %67, %out : f32
      linalg.yield %68 : f32
    }
    %44 = memref.load %alloc_105[] : memref<f32>
    %45 = arith.divf %44, %cst_3 : f32
    %46 = arith.addf %45, %cst_4 : f32
    %47 = math.rsqrt %46 : f32
    linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%alloc_98, %subview_104 : memref<768xf32>, memref<768xf32, strided<[1], offset: 3072>>) outs(%alloc_12 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %47 : f32
      %68 = arith.mulf %67, %in_151 : f32
      linalg.yield %68 : f32
    }
    %subview_106 = memref.subview %arg6[4, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 2359296>>
    %subview_107 = memref.subview %arg7[4, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 2359296>>
    %subview_108 = memref.subview %arg8[4, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 2359296>>
    %alloc_109 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc_11, %alloc_109 : memref<768xf32> to memref<768xf32>
    linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_106, %alloc_12 : memref<768x768xf32, strided<[768, 1], offset: 2359296>>, memref<768xf32>) outs(%alloc_109 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %in_151 : f32
      %68 = arith.addf %out, %67 : f32
      linalg.yield %68 : f32
    }
    %alloc_110 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc_11, %alloc_110 : memref<768xf32> to memref<768xf32>
    linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_107, %alloc_12 : memref<768x768xf32, strided<[768, 1], offset: 2359296>>, memref<768xf32>) outs(%alloc_110 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %in_151 : f32
      %68 = arith.addf %out, %67 : f32
      linalg.yield %68 : f32
    }
    %alloc_111 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc_11, %alloc_111 : memref<768xf32> to memref<768xf32>
    linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_108, %alloc_12 : memref<768x768xf32, strided<[768, 1], offset: 2359296>>, memref<768xf32>) outs(%alloc_111 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %in_151 : f32
      %68 = arith.addf %out, %67 : f32
      linalg.yield %68 : f32
    }
    %subview_112 = memref.subview %arg3[4, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
    memref.copy %alloc_111, %subview_112 : memref<768xf32> to memref<768xf32, strided<[1], offset: ?>>
    scf.for %arg17 = %c0 to %c768 step %c2 {
      %67 = arith.remui %arg17, %c48 : index
      %68 = arith.index_cast %67 : index to i64
      %69 = arith.uitofp %68 : i64 to f32
      %70 = arith.divf %69, %cst_6 : f32
      %71 = math.powf %cst_7, %70 : f32
      %72 = arith.divf %cst_5, %71 : f32
      %73 = arith.mulf %6, %72 : f32
      %74 = math.cos %73 : f32
      %75 = math.sin %73 : f32
      %76 = arith.addi %arg17, %c1 : index
      %77 = memref.load %alloc_109[%arg17] : memref<768xf32>
      %78 = memref.load %alloc_109[%76] : memref<768xf32>
      %79 = arith.mulf %77, %74 : f32
      %80 = arith.mulf %78, %75 : f32
      %81 = arith.subf %79, %80 : f32
      memref.store %81, %alloc_109[%arg17] : memref<768xf32>
      %82 = arith.mulf %77, %75 : f32
      %83 = arith.mulf %78, %74 : f32
      %84 = arith.addf %82, %83 : f32
      memref.store %84, %alloc_109[%76] : memref<768xf32>
      %85 = arith.cmpi ult, %arg17, %c768 : index
      scf.if %85 {
        %86 = memref.load %alloc_110[%arg17] : memref<768xf32>
        %87 = memref.load %alloc_110[%76] : memref<768xf32>
        %88 = arith.mulf %86, %74 : f32
        %89 = arith.mulf %87, %75 : f32
        %90 = arith.subf %88, %89 : f32
        memref.store %90, %alloc_110[%arg17] : memref<768xf32>
        %91 = arith.mulf %86, %75 : f32
        %92 = arith.mulf %87, %74 : f32
        %93 = arith.addf %91, %92 : f32
        memref.store %93, %alloc_110[%76] : memref<768xf32>
      }
    }
    %subview_113 = memref.subview %arg2[4, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
    memref.copy %alloc_110, %subview_113 : memref<768xf32> to memref<768xf32, strided<[1], offset: ?>>
    %subview_114 = memref.subview %arg2[4, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3145728>>
    %subview_115 = memref.subview %arg3[4, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3145728>>
    %alloc_116 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc_109, %alloc_116 : memref<768xf32> to memref<768xf32>
    scf.for %arg17 = %c0 to %c6 step %c1 {
      %67 = arith.muli %arg17, %c48 : index
      %alloc_151 = memref.alloc() {alignment = 64 : i64} : memref<1024xf32>
      scf.for %arg18 = %c0 to %7 step %c1 {
        %subview_154 = memref.subview %alloc_109[%67] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: ?>>
        %subview_155 = memref.subview %subview_114[%arg18, %67] [1, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<48xf32, strided<[1], offset: ?>>
        %alloc_156 = memref.alloc() {alignment = 64 : i64} : memref<f32>
        %71 = memref.load %alloc_9[] : memref<f32>
        memref.store %71, %alloc_156[] : memref<f32>
        linalg.generic {indexing_maps = [#map1, #map1, #map2], iterator_types = ["reduction"]} ins(%subview_154, %subview_155 : memref<48xf32, strided<[1], offset: ?>>, memref<48xf32, strided<[1], offset: ?>>) outs(%alloc_156 : memref<f32>) {
        ^bb0(%in: f32, %in_157: f32, %out: f32):
          %74 = arith.mulf %in, %in_157 : f32
          %75 = arith.addf %74, %out : f32
          linalg.yield %75 : f32
        }
        %72 = memref.load %alloc_156[] : memref<f32>
        %73 = arith.divf %72, %cst_2 : f32
        memref.store %73, %alloc_151[%arg18] : memref<1024xf32>
      }
      scf.for %arg18 = %7 to %c1024 step %c1 {
        memref.store %cst_1, %alloc_151[%arg18] : memref<1024xf32>
      }
      linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc : memref<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      }
      linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_151 : memref<1024xf32>) outs(%alloc : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %71 = arith.maxnumf %in, %out : f32
        linalg.yield %71 : f32
      }
      %68 = memref.load %alloc[] : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%alloc_151 : memref<1024xf32>) outs(%alloc_151 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %71 = arith.subf %in, %68 : f32
        %72 = math.exp %71 : f32
        linalg.yield %72 : f32
      }
      %alloc_152 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      %69 = memref.load %alloc_9[] : memref<f32>
      memref.store %69, %alloc_152[] : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_151 : memref<1024xf32>) outs(%alloc_152 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %71 = arith.addf %in, %out : f32
        linalg.yield %71 : f32
      }
      %70 = memref.load %alloc_152[] : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%alloc_151 : memref<1024xf32>) outs(%alloc_151 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %71 = arith.divf %in, %70 : f32
        linalg.yield %71 : f32
      }
      %subview_153 = memref.subview %alloc_116[%67] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: ?>>
      linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%subview_153 : memref<48xf32, strided<[1], offset: ?>>) {
      ^bb0(%out: f32):
        linalg.yield %cst_0 : f32
      }
      scf.for %arg18 = %c0 to %7 step %c1 {
        %subview_154 = memref.subview %subview_115[%arg18, %67] [1, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<48xf32, strided<[1], offset: ?>>
        %71 = memref.load %alloc_151[%arg18] : memref<1024xf32>
        linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%subview_153, %subview_154 : memref<48xf32, strided<[1], offset: ?>>, memref<48xf32, strided<[1], offset: ?>>) outs(%subview_153 : memref<48xf32, strided<[1], offset: ?>>) {
        ^bb0(%in: f32, %in_155: f32, %out: f32):
          %72 = arith.mulf %in_155, %71 : f32
          %73 = arith.addf %in, %72 : f32
          linalg.yield %73 : f32
        }
      }
    }
    memref.copy %alloc_116, %alloc_109 : memref<768xf32> to memref<768xf32>
    %subview_117 = memref.subview %arg9[4, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 2359296>>
    %alloc_118 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc_109, %alloc_118 : memref<768xf32> to memref<768xf32>
    linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_117, %alloc_109 : memref<768x768xf32, strided<[768, 1], offset: 2359296>>, memref<768xf32>) outs(%alloc_118 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %in_151 : f32
      %68 = arith.addf %out, %67 : f32
      linalg.yield %68 : f32
    }
    linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%alloc_98, %alloc_118 : memref<768xf32>, memref<768xf32>) outs(%alloc_118 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.addf %in, %in_151 : f32
      linalg.yield %67 : f32
    }
    %subview_119 = memref.subview %arg13[4, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: 3072>>
    %alloc_120 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    %48 = memref.load %alloc_9[] : memref<f32>
    memref.store %48, %alloc_120[] : memref<f32>
    linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_118 : memref<768xf32>) outs(%alloc_120 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %67 = arith.mulf %in, %in : f32
      %68 = arith.addf %67, %out : f32
      linalg.yield %68 : f32
    }
    %49 = memref.load %alloc_120[] : memref<f32>
    %50 = arith.divf %49, %cst_3 : f32
    %51 = arith.addf %50, %cst_4 : f32
    %52 = math.rsqrt %51 : f32
    %alloc_121 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%alloc_118, %subview_119 : memref<768xf32>, memref<768xf32, strided<[1], offset: 3072>>) outs(%alloc_121 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %52 : f32
      %68 = arith.mulf %67, %in_151 : f32
      linalg.yield %68 : f32
    }
    %subview_122 = memref.subview %arg10[4, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: 6291456>>
    %subview_123 = memref.subview %arg12[4, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: 6291456>>
    %alloc_124 = memref.alloc() {alignment = 64 : i64} : memref<2048xf32>
    memref.copy %alloc_31, %alloc_124 : memref<2048xf32> to memref<2048xf32>
    linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_122, %alloc_121 : memref<2048x768xf32, strided<[768, 1], offset: 6291456>>, memref<768xf32>) outs(%alloc_124 : memref<2048xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %in_151 : f32
      %68 = arith.addf %out, %67 : f32
      linalg.yield %68 : f32
    }
    %alloc_125 = memref.alloc() {alignment = 64 : i64} : memref<2048xf32>
    memref.copy %alloc_31, %alloc_125 : memref<2048xf32> to memref<2048xf32>
    linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_123, %alloc_121 : memref<2048x768xf32, strided<[768, 1], offset: 6291456>>, memref<768xf32>) outs(%alloc_125 : memref<2048xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %in_151 : f32
      %68 = arith.addf %out, %67 : f32
      linalg.yield %68 : f32
    }
    linalg.map ins(%alloc_124, %alloc_125 : memref<2048xf32>, memref<2048xf32>) outs(%alloc_124 : memref<2048xf32>)
      (%in: f32, %in_151: f32) {
        %67 = arith.negf %in : f32
        %68 = math.exp %67 : f32
        %69 = arith.addf %68, %cst_5 : f32
        %70 = arith.divf %cst_5, %69 : f32
        %71 = arith.mulf %in, %70 : f32
        %72 = arith.mulf %71, %in_151 : f32
        linalg.yield %72 : f32
      }
    %subview_126 = memref.subview %arg11[4, 0, 0] [1, 768, 2048] [1, 1, 1] : memref<6x768x2048xf32> to memref<768x2048xf32, strided<[2048, 1], offset: 6291456>>
    linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_126, %alloc_124 : memref<768x2048xf32, strided<[2048, 1], offset: 6291456>>, memref<2048xf32>) outs(%alloc_121 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %in_151 : f32
      %68 = arith.addf %out, %67 : f32
      linalg.yield %68 : f32
    }
    %subview_127 = memref.subview %arg5[5, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: 3840>>
    %alloc_128 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    %53 = memref.load %alloc_9[] : memref<f32>
    memref.store %53, %alloc_128[] : memref<f32>
    linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_121 : memref<768xf32>) outs(%alloc_128 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %67 = arith.mulf %in, %in : f32
      %68 = arith.addf %67, %out : f32
      linalg.yield %68 : f32
    }
    %54 = memref.load %alloc_128[] : memref<f32>
    %55 = arith.divf %54, %cst_3 : f32
    %56 = arith.addf %55, %cst_4 : f32
    %57 = math.rsqrt %56 : f32
    linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%alloc_121, %subview_127 : memref<768xf32>, memref<768xf32, strided<[1], offset: 3840>>) outs(%alloc_12 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %57 : f32
      %68 = arith.mulf %67, %in_151 : f32
      linalg.yield %68 : f32
    }
    %subview_129 = memref.subview %arg6[5, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 2949120>>
    %subview_130 = memref.subview %arg7[5, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 2949120>>
    %subview_131 = memref.subview %arg8[5, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 2949120>>
    %alloc_132 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc_11, %alloc_132 : memref<768xf32> to memref<768xf32>
    linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_129, %alloc_12 : memref<768x768xf32, strided<[768, 1], offset: 2949120>>, memref<768xf32>) outs(%alloc_132 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %in_151 : f32
      %68 = arith.addf %out, %67 : f32
      linalg.yield %68 : f32
    }
    %alloc_133 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc_11, %alloc_133 : memref<768xf32> to memref<768xf32>
    linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_130, %alloc_12 : memref<768x768xf32, strided<[768, 1], offset: 2949120>>, memref<768xf32>) outs(%alloc_133 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %in_151 : f32
      %68 = arith.addf %out, %67 : f32
      linalg.yield %68 : f32
    }
    linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_131, %alloc_12 : memref<768x768xf32, strided<[768, 1], offset: 2949120>>, memref<768xf32>) outs(%alloc_11 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %in_151 : f32
      %68 = arith.addf %out, %67 : f32
      linalg.yield %68 : f32
    }
    %subview_134 = memref.subview %arg3[5, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
    memref.copy %alloc_11, %subview_134 : memref<768xf32> to memref<768xf32, strided<[1], offset: ?>>
    scf.for %arg17 = %c0 to %c768 step %c2 {
      %67 = arith.remui %arg17, %c48 : index
      %68 = arith.index_cast %67 : index to i64
      %69 = arith.uitofp %68 : i64 to f32
      %70 = arith.divf %69, %cst_6 : f32
      %71 = math.powf %cst_7, %70 : f32
      %72 = arith.divf %cst_5, %71 : f32
      %73 = arith.mulf %6, %72 : f32
      %74 = math.cos %73 : f32
      %75 = math.sin %73 : f32
      %76 = arith.addi %arg17, %c1 : index
      %77 = memref.load %alloc_132[%arg17] : memref<768xf32>
      %78 = memref.load %alloc_132[%76] : memref<768xf32>
      %79 = arith.mulf %77, %74 : f32
      %80 = arith.mulf %78, %75 : f32
      %81 = arith.subf %79, %80 : f32
      memref.store %81, %alloc_132[%arg17] : memref<768xf32>
      %82 = arith.mulf %77, %75 : f32
      %83 = arith.mulf %78, %74 : f32
      %84 = arith.addf %82, %83 : f32
      memref.store %84, %alloc_132[%76] : memref<768xf32>
      %85 = arith.cmpi ult, %arg17, %c768 : index
      scf.if %85 {
        %86 = memref.load %alloc_133[%arg17] : memref<768xf32>
        %87 = memref.load %alloc_133[%76] : memref<768xf32>
        %88 = arith.mulf %86, %74 : f32
        %89 = arith.mulf %87, %75 : f32
        %90 = arith.subf %88, %89 : f32
        memref.store %90, %alloc_133[%arg17] : memref<768xf32>
        %91 = arith.mulf %86, %75 : f32
        %92 = arith.mulf %87, %74 : f32
        %93 = arith.addf %91, %92 : f32
        memref.store %93, %alloc_133[%76] : memref<768xf32>
      }
    }
    %subview_135 = memref.subview %arg2[5, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
    memref.copy %alloc_133, %subview_135 : memref<768xf32> to memref<768xf32, strided<[1], offset: ?>>
    %subview_136 = memref.subview %arg2[5, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3932160>>
    %subview_137 = memref.subview %arg3[5, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3932160>>
    %alloc_138 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc_132, %alloc_138 : memref<768xf32> to memref<768xf32>
    scf.for %arg17 = %c0 to %c6 step %c1 {
      %67 = arith.muli %arg17, %c48 : index
      %alloc_151 = memref.alloc() {alignment = 64 : i64} : memref<1024xf32>
      scf.for %arg18 = %c0 to %7 step %c1 {
        %subview_154 = memref.subview %alloc_132[%67] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: ?>>
        %subview_155 = memref.subview %subview_136[%arg18, %67] [1, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<48xf32, strided<[1], offset: ?>>
        %alloc_156 = memref.alloc() {alignment = 64 : i64} : memref<f32>
        %71 = memref.load %alloc_9[] : memref<f32>
        memref.store %71, %alloc_156[] : memref<f32>
        linalg.generic {indexing_maps = [#map1, #map1, #map2], iterator_types = ["reduction"]} ins(%subview_154, %subview_155 : memref<48xf32, strided<[1], offset: ?>>, memref<48xf32, strided<[1], offset: ?>>) outs(%alloc_156 : memref<f32>) {
        ^bb0(%in: f32, %in_157: f32, %out: f32):
          %74 = arith.mulf %in, %in_157 : f32
          %75 = arith.addf %74, %out : f32
          linalg.yield %75 : f32
        }
        %72 = memref.load %alloc_156[] : memref<f32>
        %73 = arith.divf %72, %cst_2 : f32
        memref.store %73, %alloc_151[%arg18] : memref<1024xf32>
      }
      scf.for %arg18 = %7 to %c1024 step %c1 {
        memref.store %cst_1, %alloc_151[%arg18] : memref<1024xf32>
      }
      linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc : memref<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      }
      linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_151 : memref<1024xf32>) outs(%alloc : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %71 = arith.maxnumf %in, %out : f32
        linalg.yield %71 : f32
      }
      %68 = memref.load %alloc[] : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%alloc_151 : memref<1024xf32>) outs(%alloc_151 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %71 = arith.subf %in, %68 : f32
        %72 = math.exp %71 : f32
        linalg.yield %72 : f32
      }
      %alloc_152 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      %69 = memref.load %alloc_9[] : memref<f32>
      memref.store %69, %alloc_152[] : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_151 : memref<1024xf32>) outs(%alloc_152 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %71 = arith.addf %in, %out : f32
        linalg.yield %71 : f32
      }
      %70 = memref.load %alloc_152[] : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%alloc_151 : memref<1024xf32>) outs(%alloc_151 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %71 = arith.divf %in, %70 : f32
        linalg.yield %71 : f32
      }
      %subview_153 = memref.subview %alloc_138[%67] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: ?>>
      linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%subview_153 : memref<48xf32, strided<[1], offset: ?>>) {
      ^bb0(%out: f32):
        linalg.yield %cst_0 : f32
      }
      scf.for %arg18 = %c0 to %7 step %c1 {
        %subview_154 = memref.subview %subview_137[%arg18, %67] [1, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<48xf32, strided<[1], offset: ?>>
        %71 = memref.load %alloc_151[%arg18] : memref<1024xf32>
        linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%subview_153, %subview_154 : memref<48xf32, strided<[1], offset: ?>>, memref<48xf32, strided<[1], offset: ?>>) outs(%subview_153 : memref<48xf32, strided<[1], offset: ?>>) {
        ^bb0(%in: f32, %in_155: f32, %out: f32):
          %72 = arith.mulf %in_155, %71 : f32
          %73 = arith.addf %in, %72 : f32
          linalg.yield %73 : f32
        }
      }
    }
    memref.copy %alloc_138, %alloc_132 : memref<768xf32> to memref<768xf32>
    %subview_139 = memref.subview %arg9[5, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 2949120>>
    %alloc_140 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc_132, %alloc_140 : memref<768xf32> to memref<768xf32>
    linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_139, %alloc_132 : memref<768x768xf32, strided<[768, 1], offset: 2949120>>, memref<768xf32>) outs(%alloc_140 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %in_151 : f32
      %68 = arith.addf %out, %67 : f32
      linalg.yield %68 : f32
    }
    linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%alloc_121, %alloc_140 : memref<768xf32>, memref<768xf32>) outs(%alloc_140 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.addf %in, %in_151 : f32
      linalg.yield %67 : f32
    }
    %subview_141 = memref.subview %arg13[5, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: 3840>>
    %alloc_142 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    %58 = memref.load %alloc_9[] : memref<f32>
    memref.store %58, %alloc_142[] : memref<f32>
    linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_140 : memref<768xf32>) outs(%alloc_142 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %67 = arith.mulf %in, %in : f32
      %68 = arith.addf %67, %out : f32
      linalg.yield %68 : f32
    }
    %59 = memref.load %alloc_142[] : memref<f32>
    %60 = arith.divf %59, %cst_3 : f32
    %61 = arith.addf %60, %cst_4 : f32
    %62 = math.rsqrt %61 : f32
    linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%alloc_140, %subview_141 : memref<768xf32>, memref<768xf32, strided<[1], offset: 3840>>) outs(%alloc_12 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %62 : f32
      %68 = arith.mulf %67, %in_151 : f32
      linalg.yield %68 : f32
    }
    %subview_143 = memref.subview %arg10[5, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: 7864320>>
    %subview_144 = memref.subview %arg12[5, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: 7864320>>
    %alloc_145 = memref.alloc() {alignment = 64 : i64} : memref<2048xf32>
    memref.copy %alloc_31, %alloc_145 : memref<2048xf32> to memref<2048xf32>
    linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_143, %alloc_12 : memref<2048x768xf32, strided<[768, 1], offset: 7864320>>, memref<768xf32>) outs(%alloc_145 : memref<2048xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %in_151 : f32
      %68 = arith.addf %out, %67 : f32
      linalg.yield %68 : f32
    }
    linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_144, %alloc_12 : memref<2048x768xf32, strided<[768, 1], offset: 7864320>>, memref<768xf32>) outs(%alloc_31 : memref<2048xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %in_151 : f32
      %68 = arith.addf %out, %67 : f32
      linalg.yield %68 : f32
    }
    linalg.map ins(%alloc_145, %alloc_31 : memref<2048xf32>, memref<2048xf32>) outs(%alloc_145 : memref<2048xf32>)
      (%in: f32, %in_151: f32) {
        %67 = arith.negf %in : f32
        %68 = math.exp %67 : f32
        %69 = arith.addf %68, %cst_5 : f32
        %70 = arith.divf %cst_5, %69 : f32
        %71 = arith.mulf %in, %70 : f32
        %72 = arith.mulf %71, %in_151 : f32
        linalg.yield %72 : f32
      }
    %subview_146 = memref.subview %arg11[5, 0, 0] [1, 768, 2048] [1, 1, 1] : memref<6x768x2048xf32> to memref<768x2048xf32, strided<[2048, 1], offset: 7864320>>
    linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_146, %alloc_145 : memref<768x2048xf32, strided<[2048, 1], offset: 7864320>>, memref<2048xf32>) outs(%alloc_12 : memref<768xf32>) {
    ^bb0(%in: f32, %in_151: f32, %out: f32):
      %67 = arith.mulf %in, %in_151 : f32
      %68 = arith.addf %out, %67 : f32
      linalg.yield %68 : f32
    }
    linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_12 : memref<768xf32>) outs(%alloc_9 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %67 = arith.mulf %in, %in : f32
      %68 = arith.addf %67, %out : f32
      linalg.yield %68 : f32
    }
    %63 = memref.load %alloc_9[] : memref<f32>
    %64 = arith.divf %63, %cst_3 : f32
    %65 = arith.addf %64, %cst_4 : f32
    %66 = math.rsqrt %65 : f32
    %alloc_147 = memref.alloc() {alignment = 64 : i64} : memref<34048x768xf32>
    linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%alloc_147 : memref<34048x768xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_0 : f32
    }
    %subview_148 = memref.subview %alloc_147[0, 0] [32000, 768] [1, 1] : memref<34048x768xf32> to memref<32000x768xf32, strided<[768, 1]>>
    memref.copy %arg15, %subview_148 : memref<32000x768xf32> to memref<32000x768xf32, strided<[768, 1]>>
    %alloc_149 = memref.alloc() {alignment = 64 : i64} : memref<34048xf32>
    linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%alloc_149 : memref<34048xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_0 : f32
    }
    linalg.generic {indexing_maps = [#map3, #map4, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%alloc_147, %alloc_12, %arg14 : memref<34048x768xf32>, memref<768xf32>, memref<768xf32>) outs(%alloc_149 : memref<34048xf32>) {
    ^bb0(%in: f32, %in_151: f32, %in_152: f32, %out: f32):
      %67 = arith.mulf %in_151, %66 : f32
      %68 = arith.mulf %67, %in_152 : f32
      %69 = arith.mulf %in, %68 : f32
      %70 = arith.addf %out, %69 : f32
      linalg.yield %70 : f32
    }
    %subview_150 = memref.subview %alloc_149[0] [32000] [1] : memref<34048xf32> to memref<32000xf32, strided<[1]>>
    memref.copy %subview_150, %arg16 : memref<32000xf32, strided<[1]>> to memref<32000xf32, strided<[1]>>
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
    %cst = arith.constant 0xFFC00000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    %c48 = arith.constant 48 : index
    %c1024 = arith.constant 1024 : index
    %cst_1 = arith.constant 6.92820311 : f32
    %cst_2 = arith.constant 0xFF800000 : f32
    %0 = arith.addi %arg3, %c1 : index
    scf.for %arg5 = %c0 to %c6 step %c1 {
      %1 = arith.muli %arg5, %c48 : index
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<1024xf32>
      scf.for %arg6 = %c0 to %0 step %c1 {
        %subview_4 = memref.subview %arg0[%1] [48] [1] : memref<768xf32, strided<[?], offset: ?>> to memref<48xf32, strided<[?], offset: ?>>
        %subview_5 = memref.subview %arg1[%arg6, %1] [1, 48] [1, 1] : memref<1024x768xf32, strided<[?, ?], offset: ?>> to memref<48xf32, strided<[?], offset: ?>>
        %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<f32>
        linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_6 : memref<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_0 : f32
        }
        linalg.generic {indexing_maps = [#map1, #map1, #map2], iterator_types = ["reduction"]} ins(%subview_4, %subview_5 : memref<48xf32, strided<[?], offset: ?>>, memref<48xf32, strided<[?], offset: ?>>) outs(%alloc_6 : memref<f32>) {
        ^bb0(%in: f32, %in_7: f32, %out: f32):
          %6 = arith.mulf %in, %in_7 : f32
          %7 = arith.addf %6, %out : f32
          linalg.yield %7 : f32
        }
        %4 = memref.load %alloc_6[] : memref<f32>
        %5 = arith.divf %4, %cst_1 : f32
        memref.store %5, %alloc[%arg6] : memref<1024xf32>
      }
      scf.for %arg6 = %0 to %c1024 step %c1 {
        memref.store %cst_2, %alloc[%arg6] : memref<1024xf32>
      }
      %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_3 : memref<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      }
      linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc : memref<1024xf32>) outs(%alloc_3 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %4 = arith.maxnumf %in, %out : f32
        linalg.yield %4 : f32
      }
      %2 = memref.load %alloc_3[] : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%alloc : memref<1024xf32>) outs(%alloc : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %4 = arith.subf %in, %2 : f32
        %5 = math.exp %4 : f32
        linalg.yield %5 : f32
      }
      linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_3 : memref<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_0 : f32
      }
      linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc : memref<1024xf32>) outs(%alloc_3 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %4 = arith.addf %in, %out : f32
        linalg.yield %4 : f32
      }
      %3 = memref.load %alloc_3[] : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%alloc : memref<1024xf32>) outs(%alloc : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %4 = arith.divf %in, %3 : f32
        linalg.yield %4 : f32
      }
      %subview = memref.subview %arg4[%1] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: ?>>
      linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%subview : memref<48xf32, strided<[1], offset: ?>>) {
      ^bb0(%out: f32):
        linalg.yield %cst_0 : f32
      }
      scf.for %arg6 = %c0 to %0 step %c1 {
        %subview_4 = memref.subview %arg2[%arg6, %1] [1, 48] [1, 1] : memref<1024x768xf32, strided<[?, ?], offset: ?>> to memref<48xf32, strided<[?], offset: ?>>
        %4 = memref.load %alloc[%arg6] : memref<1024xf32>
        linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%subview, %subview_4 : memref<48xf32, strided<[1], offset: ?>>, memref<48xf32, strided<[?], offset: ?>>) outs(%subview : memref<48xf32, strided<[1], offset: ?>>) {
        ^bb0(%in: f32, %in_5: f32, %out: f32):
          %5 = arith.mulf %in_5, %4 : f32
          %6 = arith.addf %in, %5 : f32
          linalg.yield %6 : f32
        }
      }
    }
    return
  }
  func.func @rmsnorm(%arg0: memref<768xf32, strided<[?], offset: ?>>, %arg1: memref<768xf32, strided<[?], offset: ?>>, %arg2: memref<768xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 9.99999974E-6 : f32
    %cst_1 = arith.constant 7.680000e+02 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<f32>
    linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc : memref<f32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%arg0 : memref<768xf32, strided<[?], offset: ?>>) outs(%alloc : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %4 = arith.mulf %in, %in : f32
      %5 = arith.addf %4, %out : f32
      linalg.yield %5 : f32
    }
    %0 = memref.load %alloc[] : memref<f32>
    %1 = arith.divf %0, %cst_1 : f32
    %2 = arith.addf %1, %cst_0 : f32
    %3 = math.rsqrt %2 : f32
    linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<768xf32, strided<[?], offset: ?>>, memref<768xf32, strided<[?], offset: ?>>) outs(%arg2 : memref<768xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %4 = arith.mulf %in, %3 : f32
      %5 = arith.mulf %4, %in_2 : f32
      linalg.yield %5 : f32
    }
    return
  }
  func.func @softmax(%arg0: memref<1024xf32, strided<[?], offset: ?>>, %arg1: memref<1024xf32, strided<[?], offset: ?>>) {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFFC00000 : f32
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
