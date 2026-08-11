#map = affine_map<() -> ()>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> ()>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d1)>
#map5 = affine_map<(d0, d1) -> (d0)>
#map6 = affine_map<(d0, d1) -> ()>
#upmem = #upmem.platform<type = v1A, dimensions = 40x64x24>
module {
  func.func @forward(%arg0: index, %arg1: index, %arg2: memref<6x1024x768xf32>, %arg3: memref<6x1024x768xf32>, %arg4: memref<32000x768xf32> {cinm.static}, %arg5: memref<6x768xf32> {cinm.static}, %arg6: memref<6x768x768xf32> {cinm.static}, %arg7: memref<6x768x768xf32> {cinm.static}, %arg8: memref<6x768x768xf32> {cinm.static}, %arg9: memref<6x768x768xf32> {cinm.static}, %arg10: memref<6x2048x768xf32> {cinm.static}, %arg11: memref<6x768x2048xf32> {cinm.static}, %arg12: memref<6x2048x768xf32> {cinm.static}, %arg13: memref<6x768xf32> {cinm.static}, %arg14: memref<768xf32> {cinm.static}, %arg15: memref<32000x768xf32> {cinm.static}) -> memref<32000xf32> attributes {cinm.available_platforms = [#upmem]} {
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
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2048xf32>
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<1024xf32>
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<6x1024x768xf32>
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    %subview = memref.subview %arg4[%arg0, 0] [1, 768] [1, 1] : memref<32000x768xf32> to memref<768xf32, strided<[1], offset: ?>>
    scf.for %arg16 = %c0 to %c6 step %c1 {
      %4 = cinm.compute_block (%arg17 = %subview : memref<768xf32, strided<[1], offset: ?>>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
        %cst_15 = arith.constant 0.000000e+00 : f32
        %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<f32>
        linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_16 : memref<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_15 : f32
        }
        linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%arg17 : memref<768xf32, strided<[1], offset: ?>>) outs(%alloc_16 : memref<f32>) {
        ^bb0(%in: f32, %out: f32):
          %17 = arith.mulf %in, %in : f32
          %18 = arith.addf %17, %out : f32
          linalg.yield %18 : f32
        }
        %16 = memref.load %alloc_16[] : memref<f32>
        cinm.yield %16 : f32
      }
      %5 = arith.divf %4, %cst_3 : f32
      %6 = arith.addf %5, %cst_2 : f32
      %7 = math.rsqrt %6 : f32
      cinm.compute_block (%arg17 = %arg5 : memref<6x768xf32>, %arg18 = %arg16 : index, %arg19 = %subview : memref<768xf32, strided<[1], offset: ?>>, %arg20 = %7 : f32, %arg21 = %alloc_9 : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
        %subview_15 = memref.subview %arg17[%arg18, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: ?>>
        linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%arg19, %arg20, %subview_15 : memref<768xf32, strided<[1], offset: ?>>, f32, memref<768xf32, strided<[1], offset: ?>>) outs(%arg21 : memref<768xf32>) {
        ^bb0(%in: f32, %in_16: f32, %in_17: f32, %out: f32):
          %16 = arith.mulf %in, %in_16 : f32
          %17 = arith.mulf %16, %in_17 : f32
          linalg.yield %17 : f32
        }
        cinm.yield
      }
      cinm.compute_block (%arg17 = %arg6 : memref<6x768x768xf32>, %arg18 = %arg16 : index, %arg19 = %alloc_9 : memref<768xf32>, %arg20 = %alloc_9 : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
        %cst_15 = arith.constant 0.000000e+00 : f32
        %subview_16 = memref.subview %arg17[%arg18, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: ?>>
        linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%arg19 : memref<768xf32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_15 : f32
        }
        linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_16, %arg20 : memref<768x768xf32, strided<[768, 1], offset: ?>>, memref<768xf32>) outs(%arg19 : memref<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
        ^bb0(%in: f32, %in_17: f32, %out: f32):
          %16 = arith.mulf %in, %in_17 : f32
          %17 = arith.addf %out, %16 : f32
          linalg.yield %17 : f32
        }
        cinm.yield
      }
      %8 = cinm.compute_block (%arg17 = %arg7 : memref<6x768x768xf32>, %arg18 = %arg16 : index, %arg19 = %alloc_8 : memref<6x1024x768xf32>, %arg20 = %arg1 : index, %arg21 = %alloc_9 : memref<768xf32>) -> memref<768xf32, strided<[1], offset: ?>> attributes {cinm.available_platforms = [#upmem]} {
        %cst_15 = arith.constant 0.000000e+00 : f32
        %subview_16 = memref.subview %arg17[%arg18, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: ?>>
        %subview_17 = memref.subview %arg19[%arg18, %arg20, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
        linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%subview_17 : memref<768xf32, strided<[1], offset: ?>>) {
        ^bb0(%out: f32):
          linalg.yield %cst_15 : f32
        }
        linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_16, %arg21 : memref<768x768xf32, strided<[768, 1], offset: ?>>, memref<768xf32>) outs(%subview_17 : memref<768xf32, strided<[1], offset: ?>>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
        ^bb0(%in: f32, %in_18: f32, %out: f32):
          %16 = arith.mulf %in, %in_18 : f32
          %17 = arith.addf %out, %16 : f32
          linalg.yield %17 : f32
        }
        cinm.yield %subview_17 : memref<768xf32, strided<[1], offset: ?>>
      }
      cinm.compute_block (%arg17 = %arg8 : memref<6x768x768xf32>, %arg18 = %arg16 : index, %arg19 = %arg3 : memref<6x1024x768xf32>, %arg20 = %arg1 : index, %arg21 = %alloc_9 : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
        %cst_15 = arith.constant 0.000000e+00 : f32
        %subview_16 = memref.subview %arg17[%arg18, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: ?>>
        %subview_17 = memref.subview %arg19[%arg18, %arg20, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
        linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%subview_17 : memref<768xf32, strided<[1], offset: ?>>) {
        ^bb0(%out: f32):
          linalg.yield %cst_15 : f32
        }
        linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_16, %arg21 : memref<768x768xf32, strided<[768, 1], offset: ?>>, memref<768xf32>) outs(%subview_17 : memref<768xf32, strided<[1], offset: ?>>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
        ^bb0(%in: f32, %in_18: f32, %out: f32):
          %16 = arith.mulf %in, %in_18 : f32
          %17 = arith.addf %out, %16 : f32
          linalg.yield %17 : f32
        }
        cinm.yield
      }
      %9 = arith.index_cast %arg1 : index to i64
      %10 = arith.uitofp %9 : i64 to f32
      scf.for %arg17 = %c0 to %c768 step %c2 {
        %16 = arith.remui %arg17, %c48 : index
        %17 = arith.index_cast %16 : index to i64
        %18 = arith.uitofp %17 : i64 to f32
        %19 = arith.divf %18, %cst_0 : f32
        %20 = math.powf %cst, %19 : f32
        %21 = arith.divf %cst_1, %20 : f32
        %22 = arith.mulf %10, %21 : f32
        %23 = math.cos %22 : f32
        %24 = math.sin %22 : f32
        %25 = arith.addi %arg17, %c1 : index
        %26 = memref.load %alloc_9[%arg17] : memref<768xf32>
        %27 = memref.load %alloc_9[%25] : memref<768xf32>
        %28 = arith.mulf %26, %23 : f32
        %29 = arith.mulf %27, %24 : f32
        %30 = arith.subf %28, %29 : f32
        memref.store %30, %alloc_9[%arg17] : memref<768xf32>
        %31 = arith.mulf %26, %24 : f32
        %32 = arith.mulf %27, %23 : f32
        %33 = arith.addf %31, %32 : f32
        memref.store %33, %alloc_9[%25] : memref<768xf32>
        %34 = arith.cmpi ult, %arg17, %c768 : index
        scf.if %34 {
          %35 = memref.load %8[%arg17] : memref<768xf32, strided<[1], offset: ?>>
          %36 = memref.load %8[%25] : memref<768xf32, strided<[1], offset: ?>>
          %37 = arith.mulf %35, %23 : f32
          %38 = arith.mulf %36, %24 : f32
          %39 = arith.subf %37, %38 : f32
          memref.store %39, %8[%arg17] : memref<768xf32, strided<[1], offset: ?>>
          %40 = arith.mulf %35, %24 : f32
          %41 = arith.mulf %36, %23 : f32
          %42 = arith.addf %40, %41 : f32
          memref.store %42, %8[%25] : memref<768xf32, strided<[1], offset: ?>>
        }
      }
      %subview_14 = memref.subview %arg2[%arg16, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
      memref.copy %8, %subview_14 : memref<768xf32, strided<[1], offset: ?>> to memref<768xf32, strided<[1], offset: ?>>
      %11 = arith.addi %arg1, %c1 : index
      scf.for %arg17 = %c0 to %c768 step %c48 {
        scf.for %arg18 = %c0 to %11 step %c1 {
          %18 = cinm.compute_block (%arg19 = %alloc_9 : memref<768xf32>, %arg20 = %arg17 : index, %arg21 = %arg2 : memref<6x1024x768xf32>, %arg22 = %arg16 : index, %arg23 = %arg18 : index) -> f32 attributes {cinm.available_platforms = [#upmem]} {
            %cst_16 = arith.constant 0.000000e+00 : f32
            %subview_17 = memref.subview %arg19[%arg20] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: ?>>
            %subview_18 = memref.subview %arg21[%arg22, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: ?>>
            %subview_19 = memref.subview %subview_18[%arg23, %arg20] [1, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: ?>> to memref<48xf32, strided<[1], offset: ?>>
            %alloc_20 = memref.alloc() {alignment = 64 : i64} : memref<f32>
            linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_20 : memref<f32>) {
            ^bb0(%out: f32):
              linalg.yield %cst_16 : f32
            }
            linalg.generic {indexing_maps = [#map1, #map1, #map2], iterator_types = ["reduction"]} ins(%subview_17, %subview_19 : memref<48xf32, strided<[1], offset: ?>>, memref<48xf32, strided<[1], offset: ?>>) outs(%alloc_20 : memref<f32>) {
            ^bb0(%in: f32, %in_21: f32, %out: f32):
              %21 = arith.mulf %in, %in_21 : f32
              %22 = arith.addf %21, %out : f32
              linalg.yield %22 : f32
            }
            %20 = memref.load %alloc_20[] : memref<f32>
            cinm.yield %20 : f32
          }
          %19 = arith.divf %18, %cst_4 : f32
          memref.store %19, %alloc_7[%arg18] : memref<1024xf32>
        }
        scf.for %arg18 = %11 to %c1024 step %c1 {
          memref.store %cst_5, %alloc_7[%arg18] : memref<1024xf32>
        }
        %16 = cinm.compute_block (%arg18 = %alloc_7 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
          %cst_16 = arith.constant 0xFFC00000 : f32
          %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<f32>
          linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_17 : memref<f32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_16 : f32
          }
          linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%arg18 : memref<1024xf32>) outs(%alloc_17 : memref<f32>) {
          ^bb0(%in: f32, %out: f32):
            %19 = arith.maxnumf %in, %out : f32
            linalg.yield %19 : f32
          }
          %18 = memref.load %alloc_17[] : memref<f32>
          cinm.yield %18 : f32
        }
        cinm.compute_block (%arg18 = %alloc_7 : memref<1024xf32>, %arg19 = %16 : f32) attributes {cinm.available_platforms = [#upmem]} {
          linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%arg18, %arg19 : memref<1024xf32>, f32) outs(%arg18 : memref<1024xf32>) {
          ^bb0(%in: f32, %in_16: f32, %out: f32):
            %18 = arith.subf %in, %in_16 : f32
            %19 = math.exp %18 : f32
            linalg.yield %19 : f32
          }
          cinm.yield
        }
        %17 = cinm.compute_block (%arg18 = %alloc_7 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
          %cst_16 = arith.constant 0.000000e+00 : f32
          %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<f32>
          linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_17 : memref<f32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_16 : f32
          }
          linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%arg18 : memref<1024xf32>) outs(%alloc_17 : memref<f32>) {
          ^bb0(%in: f32, %out: f32):
            %19 = arith.addf %in, %out : f32
            linalg.yield %19 : f32
          }
          %18 = memref.load %alloc_17[] : memref<f32>
          cinm.yield %18 : f32
        }
        cinm.compute_block (%arg18 = %alloc_7 : memref<1024xf32>, %arg19 = %17 : f32, %arg20 = %alloc_7 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
          linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%arg18, %arg19 : memref<1024xf32>, f32) outs(%arg20 : memref<1024xf32>) {
          ^bb0(%in: f32, %in_16: f32, %out: f32):
            %18 = arith.divf %in, %in_16 : f32
            linalg.yield %18 : f32
          }
          cinm.yield
        }
        %subview_15 = memref.subview %alloc_9[%arg17] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: ?>>
        linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%subview_15 : memref<48xf32, strided<[1], offset: ?>>) {
        ^bb0(%out: f32):
          linalg.yield %cst_6 : f32
        }
        scf.for %arg18 = %c0 to %11 step %c1 {
          %18 = memref.load %alloc_7[%arg18] : memref<1024xf32>
          cinm.compute_block (%arg19 = %alloc_9 : memref<768xf32>, %arg20 = %arg17 : index, %arg21 = %arg3 : memref<6x1024x768xf32>, %arg22 = %arg16 : index, %arg23 = %arg18 : index, %arg24 = %18 : f32) attributes {cinm.available_platforms = [#upmem]} {
            %subview_16 = memref.subview %arg19[%arg20] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: ?>>
            %subview_17 = memref.subview %arg21[%arg22, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: ?>>
            %subview_18 = memref.subview %subview_17[%arg23, %arg20] [1, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: ?>> to memref<48xf32, strided<[1], offset: ?>>
            linalg.generic {indexing_maps = [#map1, #map1, #map2, #map1], iterator_types = ["parallel"]} ins(%subview_16, %subview_18, %arg24 : memref<48xf32, strided<[1], offset: ?>>, memref<48xf32, strided<[1], offset: ?>>, f32) outs(%subview_16 : memref<48xf32, strided<[1], offset: ?>>) {
            ^bb0(%in: f32, %in_19: f32, %in_20: f32, %out: f32):
              %19 = arith.mulf %in_19, %in_20 : f32
              %20 = arith.addf %in, %19 : f32
              linalg.yield %20 : f32
            }
            cinm.yield
          }
        }
      }
      cinm.compute_block (%arg17 = %arg9 : memref<6x768x768xf32>, %arg18 = %arg16 : index, %arg19 = %alloc_9 : memref<768xf32>, %arg20 = %subview : memref<768xf32, strided<[1], offset: ?>>) attributes {cinm.available_platforms = [#upmem]} {
        %subview_15 = memref.subview %arg17[%arg18, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: ?>>
        linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_15, %arg19 : memref<768x768xf32, strided<[768, 1], offset: ?>>, memref<768xf32>) outs(%arg20 : memref<768xf32, strided<[1], offset: ?>>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
        ^bb0(%in: f32, %in_16: f32, %out: f32):
          %16 = arith.mulf %in, %in_16 : f32
          %17 = arith.addf %out, %16 : f32
          linalg.yield %17 : f32
        }
        cinm.yield
      }
      %12 = cinm.compute_block (%arg17 = %subview : memref<768xf32, strided<[1], offset: ?>>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
        %cst_15 = arith.constant 0.000000e+00 : f32
        %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<f32>
        linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_16 : memref<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_15 : f32
        }
        linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%arg17 : memref<768xf32, strided<[1], offset: ?>>) outs(%alloc_16 : memref<f32>) {
        ^bb0(%in: f32, %out: f32):
          %17 = arith.mulf %in, %in : f32
          %18 = arith.addf %17, %out : f32
          linalg.yield %18 : f32
        }
        %16 = memref.load %alloc_16[] : memref<f32>
        cinm.yield %16 : f32
      }
      %13 = arith.divf %12, %cst_3 : f32
      %14 = arith.addf %13, %cst_2 : f32
      %15 = math.rsqrt %14 : f32
      cinm.compute_block (%arg17 = %arg13 : memref<6x768xf32>, %arg18 = %arg16 : index, %arg19 = %subview : memref<768xf32, strided<[1], offset: ?>>, %arg20 = %15 : f32, %arg21 = %alloc_9 : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
        %subview_15 = memref.subview %arg17[%arg18, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: ?>>
        linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%arg19, %arg20, %subview_15 : memref<768xf32, strided<[1], offset: ?>>, f32, memref<768xf32, strided<[1], offset: ?>>) outs(%arg21 : memref<768xf32>) {
        ^bb0(%in: f32, %in_16: f32, %in_17: f32, %out: f32):
          %16 = arith.mulf %in, %in_16 : f32
          %17 = arith.mulf %16, %in_17 : f32
          linalg.yield %17 : f32
        }
        cinm.yield
      }
      cinm.compute_block (%arg17 = %arg10 : memref<6x2048x768xf32>, %arg18 = %arg16 : index, %arg19 = %alloc : memref<2048xf32>, %arg20 = %alloc_9 : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
        %cst_15 = arith.constant 0.000000e+00 : f32
        %subview_16 = memref.subview %arg17[%arg18, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: ?>>
        linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%arg19 : memref<2048xf32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_15 : f32
        }
        linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_16, %arg20 : memref<2048x768xf32, strided<[768, 1], offset: ?>>, memref<768xf32>) outs(%arg19 : memref<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
        ^bb0(%in: f32, %in_17: f32, %out: f32):
          %16 = arith.mulf %in, %in_17 : f32
          %17 = arith.addf %out, %16 : f32
          linalg.yield %17 : f32
        }
        cinm.yield
      }
      cinm.compute_block (%arg17 = %arg12 : memref<6x2048x768xf32>, %arg18 = %arg16 : index, %arg19 = %alloc : memref<2048xf32>, %arg20 = %alloc_9 : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
        %cst_15 = arith.constant 0.000000e+00 : f32
        %subview_16 = memref.subview %arg17[%arg18, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: ?>>
        linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%arg19 : memref<2048xf32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_15 : f32
        }
        linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_16, %arg20 : memref<2048x768xf32, strided<[768, 1], offset: ?>>, memref<768xf32>) outs(%arg19 : memref<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
        ^bb0(%in: f32, %in_17: f32, %out: f32):
          %16 = arith.mulf %in, %in_17 : f32
          %17 = arith.addf %out, %16 : f32
          linalg.yield %17 : f32
        }
        cinm.yield
      }
      cinm.compute_block (%arg17 = %arg11 : memref<6x768x2048xf32>, %arg18 = %arg16 : index, %arg19 = %alloc : memref<2048xf32>, %arg20 = %alloc : memref<2048xf32>, %arg21 = %subview : memref<768xf32, strided<[1], offset: ?>>) attributes {cinm.available_platforms = [#upmem]} {
        %cst_15 = arith.constant 1.000000e+00 : f32
        %subview_16 = memref.subview %arg17[%arg18, 0, 0] [1, 768, 2048] [1, 1, 1] : memref<6x768x2048xf32> to memref<768x2048xf32, strided<[2048, 1], offset: ?>>
        linalg.generic {indexing_maps = [#map3, #map4, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%subview_16, %arg19 : memref<768x2048xf32, strided<[2048, 1], offset: ?>>, memref<2048xf32>) outs(%arg20, %arg21 : memref<2048xf32>, memref<768xf32, strided<[1], offset: ?>>) {
        ^bb0(%in: f32, %in_17: f32, %out: f32, %out_18: f32):
          %16 = arith.negf %out : f32
          %17 = math.exp %16 : f32
          %18 = arith.addf %17, %cst_15 : f32
          %19 = arith.divf %cst_15, %18 : f32
          %20 = arith.mulf %out, %19 : f32
          %21 = arith.mulf %20, %in_17 : f32
          %22 = arith.mulf %in, %21 : f32
          %23 = arith.addf %out_18, %22 : f32
          linalg.yield %21, %23 : f32, f32
        }
        cinm.yield
      }
    }
    %0 = cinm.compute_block (%arg16 = %subview : memref<768xf32, strided<[1], offset: ?>>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst_14 = arith.constant 0.000000e+00 : f32
      %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_15 : memref<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_14 : f32
      }
      linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%arg16 : memref<768xf32, strided<[1], offset: ?>>) outs(%alloc_15 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %5 = arith.mulf %in, %in : f32
        %6 = arith.addf %5, %out : f32
        linalg.yield %6 : f32
      }
      %4 = memref.load %alloc_15[] : memref<f32>
      cinm.yield %4 : f32
    }
    %1 = arith.divf %0, %cst_3 : f32
    %2 = arith.addf %1, %cst_2 : f32
    %3 = math.rsqrt %2 : f32
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<34048x768xf32>
    linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%alloc_10 : memref<34048x768xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_6 : f32
    }
    %subview_11 = memref.subview %alloc_10[0, 0] [32000, 768] [1, 1] : memref<34048x768xf32> to memref<32000x768xf32, strided<[768, 1]>>
    memref.copy %arg15, %subview_11 : memref<32000x768xf32> to memref<32000x768xf32, strided<[768, 1]>>
    %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<34048xf32>
    cinm.compute_block (%arg16 = %alloc_12 : memref<34048xf32>, %arg17 = %alloc_10 : memref<34048x768xf32>, %arg18 = %subview : memref<768xf32, strided<[1], offset: ?>>, %arg19 = %3 : f32, %arg20 = %arg14 : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %cst_14 = arith.constant 0.000000e+00 : f32
      linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%arg16 : memref<34048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_14 : f32
      }
      linalg.generic {indexing_maps = [#map3, #map4, #map6, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18, %arg19, %arg20 : memref<34048x768xf32>, memref<768xf32, strided<[1], offset: ?>>, f32, memref<768xf32>) outs(%arg16 : memref<34048xf32>) {
      ^bb0(%in: f32, %in_15: f32, %in_16: f32, %in_17: f32, %out: f32):
        %4 = arith.mulf %in_15, %in_16 : f32
        %5 = arith.mulf %4, %in_17 : f32
        %6 = arith.mulf %in, %5 : f32
        %7 = arith.addf %out, %6 : f32
        linalg.yield %7 : f32
      }
      cinm.yield
    }
    %subview_13 = memref.subview %alloc_12[0] [32000] [1] : memref<34048xf32> to memref<32000xf32, strided<[1]>>
    %cast = memref.cast %subview_13 : memref<32000xf32, strided<[1]>> to memref<32000xf32>
    return %cast : memref<32000xf32>
  }
  func.func @rot(%arg0: memref<768xf32>, %arg1: index, %arg2: f32, %arg3: f32) -> memref<768xf32> {
    %c1 = arith.constant 1 : index
    %0 = arith.addi %arg1, %c1 : index
    %1 = memref.load %arg0[%arg1] : memref<768xf32>
    %2 = memref.load %arg0[%0] : memref<768xf32>
    %3 = arith.mulf %1, %arg2 : f32
    %4 = arith.mulf %2, %arg3 : f32
    %5 = arith.subf %3, %4 : f32
    memref.store %5, %arg0[%arg1] : memref<768xf32>
    %6 = arith.mulf %1, %arg3 : f32
    %7 = arith.mulf %2, %arg2 : f32
    %8 = arith.addf %6, %7 : f32
    memref.store %8, %arg0[%0] : memref<768xf32>
    return %arg0 : memref<768xf32>
  }
  func.func @mha(%arg0: memref<768xf32>, %arg1: memref<1024x768xf32>, %arg2: memref<1024x768xf32>, %arg3: index) -> memref<768xf32> {
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 6.92820311 : f32
    %c1024 = arith.constant 1024 : index
    %c768 = arith.constant 768 : index
    %c48 = arith.constant 48 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 0xFFC00000 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<f32>
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<1024xf32>
    %0 = arith.addi %arg3, %c1 : index
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    scf.for %arg4 = %c0 to %c768 step %c48 {
      scf.for %arg5 = %c0 to %0 step %c1 {
        %subview_6 = memref.subview %arg0[%arg4] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: ?>>
        %subview_7 = memref.subview %arg1[%arg5, %arg4] [1, 48] [1, 1] : memref<1024x768xf32> to memref<48xf32, strided<[1], offset: ?>>
        linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc_3 : memref<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_1 : f32
        }
        linalg.generic {indexing_maps = [#map1, #map1, #map2], iterator_types = ["reduction"]} ins(%subview_6, %subview_7 : memref<48xf32, strided<[1], offset: ?>>, memref<48xf32, strided<[1], offset: ?>>) outs(%alloc_3 : memref<f32>) {
        ^bb0(%in: f32, %in_8: f32, %out: f32):
          %5 = arith.mulf %in, %in_8 : f32
          %6 = arith.addf %5, %out : f32
          linalg.yield %6 : f32
        }
        %3 = memref.load %alloc_3[] : memref<f32>
        %4 = arith.divf %3, %cst_0 : f32
        memref.store %4, %alloc_4[%arg5] : memref<1024xf32>
      }
      scf.for %arg5 = %0 to %c1024 step %c1 {
        memref.store %cst, %alloc_4[%arg5] : memref<1024xf32>
      }
      linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc : memref<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      }
      linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_4 : memref<1024xf32>) outs(%alloc : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %3 = arith.maxnumf %in, %out : f32
        linalg.yield %3 : f32
      }
      %1 = memref.load %alloc[] : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%alloc_4, %1 : memref<1024xf32>, f32) outs(%alloc_4 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_6: f32, %out: f32):
        %3 = arith.subf %in, %in_6 : f32
        %4 = math.exp %3 : f32
        linalg.yield %4 : f32
      }
      linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc : memref<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_1 : f32
      }
      linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%alloc_4 : memref<1024xf32>) outs(%alloc : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %3 = arith.addf %in, %out : f32
        linalg.yield %3 : f32
      }
      %2 = memref.load %alloc[] : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%alloc_4, %2 : memref<1024xf32>, f32) outs(%alloc_4 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_6: f32, %out: f32):
        %3 = arith.divf %in, %in_6 : f32
        linalg.yield %3 : f32
      }
      %subview = memref.subview %alloc_5[%arg4] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: ?>>
      linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%subview : memref<48xf32, strided<[1], offset: ?>>) {
      ^bb0(%out: f32):
        linalg.yield %cst_1 : f32
      }
      scf.for %arg5 = %c0 to %0 step %c1 {
        %subview_6 = memref.subview %arg2[%arg5, %arg4] [1, 48] [1, 1] : memref<1024x768xf32> to memref<48xf32, strided<[1], offset: ?>>
        %3 = memref.load %alloc_4[%arg5] : memref<1024xf32>
        linalg.generic {indexing_maps = [#map1, #map1, #map2, #map1], iterator_types = ["parallel"]} ins(%subview, %subview_6, %3 : memref<48xf32, strided<[1], offset: ?>>, memref<48xf32, strided<[1], offset: ?>>, f32) outs(%subview : memref<48xf32, strided<[1], offset: ?>>) {
        ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
          %4 = arith.mulf %in_7, %in_8 : f32
          %5 = arith.addf %in, %4 : f32
          linalg.yield %5 : f32
        }
      }
    }
    return %alloc_5 : memref<768xf32>
  }
  func.func @rmsnorm(%arg0: memref<768xf32>, %arg1: memref<768xf32>) -> memref<768xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 9.99999974E-6 : f32
    %cst_1 = arith.constant 7.680000e+02 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<f32>
    linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc : memref<f32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%arg0 : memref<768xf32>) outs(%alloc : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %4 = arith.mulf %in, %in : f32
      %5 = arith.addf %4, %out : f32
      linalg.yield %5 : f32
    }
    %0 = memref.load %alloc[] : memref<f32>
    %1 = arith.divf %0, %cst_1 : f32
    %2 = arith.addf %1, %cst_0 : f32
    %3 = math.rsqrt %2 : f32
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%arg0, %3, %arg1 : memref<768xf32>, f32, memref<768xf32>) outs(%alloc_2 : memref<768xf32>) {
    ^bb0(%in: f32, %in_3: f32, %in_4: f32, %out: f32):
      %4 = arith.mulf %in, %in_3 : f32
      %5 = arith.mulf %4, %in_4 : f32
      linalg.yield %5 : f32
    }
    return %alloc_2 : memref<768xf32>
  }
  func.func @softmax(%arg0: memref<1024xf32>) -> memref<1024xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFFC00000 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<f32>
    linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc : memref<f32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_0 : f32
    }
    linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%arg0 : memref<1024xf32>) outs(%alloc : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.maxnumf %in, %out : f32
      linalg.yield %2 : f32
    }
    %0 = memref.load %alloc[] : memref<f32>
    linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%arg0, %0 : memref<1024xf32>, f32) outs(%arg0 : memref<1024xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.subf %in, %in_1 : f32
      %3 = math.exp %2 : f32
      linalg.yield %3 : f32
    }
    linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc : memref<f32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%arg0 : memref<1024xf32>) outs(%alloc : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.addf %in, %out : f32
      linalg.yield %2 : f32
    }
    %1 = memref.load %alloc[] : memref<f32>
    linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%arg0, %1 : memref<1024xf32>, f32) outs(%arg0 : memref<1024xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %2 = arith.divf %in, %in_1 : f32
      linalg.yield %2 : f32
    }
    return %arg0 : memref<1024xf32>
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
