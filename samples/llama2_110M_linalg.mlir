#map = affine_map<() -> ()>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> ()>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d1)>
#map5 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @forward(%arg0: index, %arg1: index, %arg2: tensor<6x1024x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, bufferization.writable = true}, %arg3: tensor<6x1024x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, bufferization.writable = true}, %arg4: tensor<32000x768xf32> {bufferization.buffer_layout = #map3}, %arg5: tensor<6x768xf32> {bufferization.buffer_layout = #map3}, %arg6: tensor<6x768x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}, %arg7: tensor<6x768x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}, %arg8: tensor<6x768x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}, %arg9: tensor<6x768x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}, %arg10: tensor<6x2048x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}, %arg11: tensor<6x768x2048xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}, %arg12: tensor<6x2048x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}, %arg13: tensor<6x768xf32> {bufferization.buffer_layout = #map3}, %arg14: tensor<768xf32> {bufferization.buffer_layout = #map1}, %arg15: tensor<32000x768xf32> {bufferization.buffer_layout = #map3}) -> tensor<32000xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 6.92820311 : f32
    %c1024 = arith.constant 1024 : index
    %cst_2 = arith.constant 7.680000e+02 : f32
    %cst_3 = arith.constant 9.99999974E-6 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c6 = arith.constant 6 : index
    %c48 = arith.constant 48 : index
    %c768 = arith.constant 768 : index
    %cst_4 = arith.constant 1.000000e+00 : f32
    %cst_5 = arith.constant 4.800000e+01 : f32
    %cst_6 = arith.constant 1.000000e+04 : f32
    %extracted_slice = tensor.extract_slice %arg4[%arg0, 0] [1, 768] [1, 1] : tensor<32000x768xf32> to tensor<768xf32>
    %0:3 = scf.for %arg16 = %c0 to %c6 step %c1 iter_args(%arg17 = %extracted_slice, %arg18 = %arg2, %arg19 = %arg3) -> (tensor<768xf32>, tensor<6x1024x768xf32>, tensor<6x1024x768xf32>) {
      %extracted_slice_7 = tensor.extract_slice %arg5[%arg16, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
      %3 = cinm.compute_ -> tensor<768xf32>
           attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
        %16 = tensor.empty() : tensor<f32>
        %17 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%16 : tensor<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst : f32
        } -> tensor<f32>
        %18 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%arg17 : tensor<768xf32>) outs(%17 : tensor<f32>) {
        ^bb0(%in: f32, %out: f32):
          %24 = arith.mulf %in, %in : f32
          %25 = arith.addf %24, %out : f32
          linalg.yield %25 : f32
        } -> tensor<f32>
        %extracted = tensor.extract %18[] : tensor<f32>
        %19 = arith.divf %extracted, %cst_2 : f32
        %20 = arith.addf %19, %cst_3 : f32
        %21 = math.rsqrt %20 : f32
        %22 = tensor.empty() : tensor<768xf32>
        %23 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%arg17, %extracted_slice_7 : tensor<768xf32>, tensor<768xf32>) outs(%22 : tensor<768xf32>) {
        ^bb0(%in: f32, %in_19: f32, %out: f32):
          %24 = arith.mulf %in, %21 : f32
          %25 = arith.mulf %24, %in_19 : f32
          linalg.yield %25 : f32
        } -> tensor<768xf32>
        cinm.yield %23 : tensor<768xf32>
      }
      %extracted_slice_8 = tensor.extract_slice %arg6[%arg16, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      %extracted_slice_9 = tensor.extract_slice %arg7[%arg16, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      %extracted_slice_10 = tensor.extract_slice %arg8[%arg16, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      %4:3 = cinm.compute_ -> tensor<768xf32>, tensor<6x1024x768xf32>, tensor<6x1024x768xf32>
           attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 6, 8>} {
        %extracted_slice_19 = tensor.extract_slice %arg18[%arg16, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
        %16 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_19 : tensor<768xf32>) {
        ^bb0(%out: f32):
          linalg.yield %cst : f32
        } -> tensor<768xf32>
        %17 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_8, %3 : tensor<768x768xf32>, tensor<768xf32>) outs(%16 : tensor<768xf32>) {
        ^bb0(%in: f32, %in_22: f32, %out: f32):
          %20 = arith.mulf %in, %in_22 : f32
          %21 = arith.addf %out, %20 : f32
          linalg.yield %21 : f32
        } -> tensor<768xf32>
        %18 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %3 : tensor<768x768xf32>, tensor<768xf32>) outs(%16 : tensor<768xf32>) {
        ^bb0(%in: f32, %in_22: f32, %out: f32):
          %20 = arith.mulf %in, %in_22 : f32
          %21 = arith.addf %out, %20 : f32
          linalg.yield %21 : f32
        } -> tensor<768xf32>
        %19 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_10, %3 : tensor<768x768xf32>, tensor<768xf32>) outs(%16 : tensor<768xf32>) {
        ^bb0(%in: f32, %in_22: f32, %out: f32):
          %20 = arith.mulf %in, %in_22 : f32
          %21 = arith.addf %out, %20 : f32
          linalg.yield %21 : f32
        } -> tensor<768xf32>
        %inserted_slice_20 = tensor.insert_slice %18 into %arg18[%arg16, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
        %inserted_slice_21 = tensor.insert_slice %19 into %arg19[%arg16, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
        cinm.yield %17, %inserted_slice_20, %inserted_slice_21 : tensor<768xf32>, tensor<6x1024x768xf32>, tensor<6x1024x768xf32>
      }
      %5 = arith.index_cast %arg1 : index to i64
      %6 = arith.uitofp %5 : i64 to f32
      %extracted_slice_11 = tensor.extract_slice %4#1[%arg16, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
      %7:2 = scf.for %arg20 = %c0 to %c768 step %c2 iter_args(%arg21 = %4#0, %arg22 = %extracted_slice_11) -> (tensor<768xf32>, tensor<768xf32>) {
        %16 = arith.remui %arg20, %c48 : index
        %17 = arith.index_cast %16 : index to i64
        %18 = arith.uitofp %17 : i64 to f32
        %19 = arith.divf %18, %cst_5 : f32
        %20 = math.powf %cst_6, %19 : f32
        %21 = arith.divf %cst_4, %20 : f32
        %22 = arith.mulf %6, %21 : f32
        %23 = math.cos %22 : f32
        %24 = math.sin %22 : f32
        %25 = arith.addi %arg20, %c1 : index
        %extracted = tensor.extract %arg21[%arg20] : tensor<768xf32>
        %extracted_19 = tensor.extract %arg21[%25] : tensor<768xf32>
        %26 = arith.mulf %extracted, %23 : f32
        %27 = arith.mulf %extracted_19, %24 : f32
        %28 = arith.subf %26, %27 : f32
        %inserted = tensor.insert %28 into %arg21[%arg20] : tensor<768xf32>
        %29 = arith.mulf %extracted, %24 : f32
        %30 = arith.mulf %extracted_19, %23 : f32
        %31 = arith.addf %29, %30 : f32
        %inserted_20 = tensor.insert %31 into %inserted[%25] : tensor<768xf32>
        %32 = bufferization.materialize_in_destination %inserted_20 in %arg21 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
        %33 = arith.cmpi ult, %arg20, %c768 : index
        %34 = scf.if %33 -> (tensor<768xf32>) {
          %extracted_21 = tensor.extract %arg22[%arg20] : tensor<768xf32>
          %extracted_22 = tensor.extract %arg22[%25] : tensor<768xf32>
          %35 = arith.mulf %extracted_21, %23 : f32
          %36 = arith.mulf %extracted_22, %24 : f32
          %37 = arith.subf %35, %36 : f32
          %inserted_23 = tensor.insert %37 into %arg22[%arg20] : tensor<768xf32>
          %38 = arith.mulf %extracted_21, %24 : f32
          %39 = arith.mulf %extracted_22, %23 : f32
          %40 = arith.addf %38, %39 : f32
          %inserted_24 = tensor.insert %40 into %inserted_23[%25] : tensor<768xf32>
          %41 = bufferization.materialize_in_destination %inserted_24 in %arg22 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
          scf.yield %41 : tensor<768xf32>
        } else {
          scf.yield %arg22 : tensor<768xf32>
        }
        scf.yield %32, %34 : tensor<768xf32>, tensor<768xf32>
      }
      %inserted_slice = tensor.insert_slice %7#1 into %4#1[%arg16, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
      %extracted_slice_12 = tensor.extract_slice %inserted_slice[%arg16, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_13 = tensor.extract_slice %4#2[%arg16, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %8 = arith.addi %arg1, %c1 : index
      %9 = tensor.empty() : tensor<768xf32>
      %10 = scf.for %arg20 = %c0 to %c6 step %c1 iter_args(%arg21 = %7#0) -> (tensor<768xf32>) {
        %16 = arith.muli %arg20, %c48 : index
        %17 = tensor.empty() : tensor<1024xf32>
        %18 = scf.for %arg22 = %c0 to %8 step %c1 iter_args(%arg23 = %17) -> (tensor<1024xf32>) {
          %extracted_slice_21 = tensor.extract_slice %7#0[%16] [48] [1] : tensor<768xf32> to tensor<48xf32>
          %extracted_slice_22 = tensor.extract_slice %extracted_slice_12[%arg22, %16] [1, 48] [1, 1] : tensor<1024x768xf32> to tensor<48xf32>
          %23 = cinm.compute_ -> f32
               attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 1, 8>} {
            %24 = tensor.empty() : tensor<f32>
            %25 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%24 : tensor<f32>) {
            ^bb0(%out: f32):
              linalg.yield %cst : f32
            } -> tensor<f32>
            %26 = linalg.generic {indexing_maps = [#map1, #map1, #map2], iterator_types = ["reduction"]} ins(%extracted_slice_21, %extracted_slice_22 : tensor<48xf32>, tensor<48xf32>) outs(%25 : tensor<f32>) {
            ^bb0(%in: f32, %in_23: f32, %out: f32):
              %28 = arith.mulf %in, %in_23 : f32
              %29 = arith.addf %28, %out : f32
              linalg.yield %29 : f32
            } -> tensor<f32>
            %extracted = tensor.extract %26[] : tensor<f32>
            %27 = arith.divf %extracted, %cst_1 : f32
            cinm.yield %27 : f32
          }
          %inserted = tensor.insert %23 into %arg23[%arg22] : tensor<1024xf32>
          scf.yield %inserted : tensor<1024xf32>
        }
        %19 = scf.for %arg22 = %8 to %c1024 step %c1 iter_args(%arg23 = %18) -> (tensor<1024xf32>) {
          %inserted = tensor.insert %cst_0 into %arg23[%arg22] : tensor<1024xf32>
          scf.yield %inserted : tensor<1024xf32>
        }
        %20 = cinm.compute_ -> tensor<1024xf32>
             attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
          %23 = tensor.empty() : tensor<f32>
          %24 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%23 : tensor<f32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_0 : f32
          } -> tensor<f32>
          %25 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%19 : tensor<1024xf32>) outs(%24 : tensor<f32>) {
          ^bb0(%in: f32, %out: f32):
            %30 = arith.maxnumf %in, %out : f32
            linalg.yield %30 : f32
          } -> tensor<f32>
          %extracted = tensor.extract %25[] : tensor<f32>
          %26 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%19 : tensor<1024xf32>) outs(%19 : tensor<1024xf32>) {
          ^bb0(%in: f32, %out: f32):
            %30 = arith.subf %in, %extracted : f32
            %31 = math.exp %30 : f32
            linalg.yield %31 : f32
          } -> tensor<1024xf32>
          %27 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%23 : tensor<f32>) {
          ^bb0(%out: f32):
            linalg.yield %cst : f32
          } -> tensor<f32>
          %28 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%26 : tensor<1024xf32>) outs(%27 : tensor<f32>) {
          ^bb0(%in: f32, %out: f32):
            %30 = arith.addf %in, %out : f32
            linalg.yield %30 : f32
          } -> tensor<f32>
          %extracted_21 = tensor.extract %28[] : tensor<f32>
          %29 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%26 : tensor<1024xf32>) outs(%19 : tensor<1024xf32>) {
          ^bb0(%in: f32, %out: f32):
            %30 = arith.divf %in, %extracted_21 : f32
            linalg.yield %30 : f32
          } -> tensor<1024xf32>
          cinm.yield %29 : tensor<1024xf32>
        }
        %extracted_slice_19 = tensor.extract_slice %arg21[%16] [48] [1] : tensor<768xf32> to tensor<48xf32>
        %21 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_19 : tensor<48xf32>) {
        ^bb0(%out: f32):
          linalg.yield %cst : f32
        } -> tensor<48xf32>
        %inserted_slice_20 = tensor.insert_slice %21 into %arg21[%16] [48] [1] : tensor<48xf32> into tensor<768xf32>
        %22 = scf.for %arg22 = %c0 to %8 step %c1 iter_args(%arg23 = %inserted_slice_20) -> (tensor<768xf32>) {
          %extracted_slice_21 = tensor.extract_slice %arg23[%16] [48] [1] : tensor<768xf32> to tensor<48xf32>
          %extracted_slice_22 = tensor.extract_slice %extracted_slice_13[%arg22, %16] [1, 48] [1, 1] : tensor<1024x768xf32> to tensor<48xf32>
          %extracted = tensor.extract %20[%arg22] : tensor<1024xf32>
          %23 = cinm.compute_ -> tensor<48xf32>
               attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 1, 8>} {
            %24 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%extracted_slice_21, %extracted_slice_22 : tensor<48xf32>, tensor<48xf32>) outs(%extracted_slice_21 : tensor<48xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %25 = arith.mulf %in_24, %extracted : f32
              %26 = arith.addf %in, %25 : f32
              linalg.yield %26 : f32
            } -> tensor<48xf32>
            cinm.yield %24 : tensor<48xf32>
          }
          %inserted_slice_23 = tensor.insert_slice %23 into %arg23[%16] [48] [1] : tensor<48xf32> into tensor<768xf32>
          scf.yield %inserted_slice_23 : tensor<768xf32>
        }
        scf.yield %22 : tensor<768xf32>
      }
      %11 = bufferization.materialize_in_destination %10 in %7#0 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      %extracted_slice_14 = tensor.extract_slice %arg9[%arg16, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      %12 = cinm.compute_ -> tensor<768xf32>
           attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 6, 8>} {
        %16 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_14, %11 : tensor<768x768xf32>, tensor<768xf32>) outs(%11 : tensor<768xf32>) {
        ^bb0(%in: f32, %in_19: f32, %out: f32):
          %18 = arith.mulf %in, %in_19 : f32
          %19 = arith.addf %out, %18 : f32
          linalg.yield %19 : f32
        } -> tensor<768xf32>
        %17 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%arg17, %16 : tensor<768xf32>, tensor<768xf32>) outs(%16 : tensor<768xf32>) {
        ^bb0(%in: f32, %in_19: f32, %out: f32):
          %18 = arith.addf %in, %in_19 : f32
          linalg.yield %18 : f32
        } -> tensor<768xf32>
        cinm.yield %17 : tensor<768xf32>
      }
      %extracted_slice_15 = tensor.extract_slice %arg13[%arg16, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
      %13 = cinm.compute_ -> tensor<768xf32>
           attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
        %16 = tensor.empty() : tensor<f32>
        %17 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%16 : tensor<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst : f32
        } -> tensor<f32>
        %18 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%12 : tensor<768xf32>) outs(%17 : tensor<f32>) {
        ^bb0(%in: f32, %out: f32):
          %23 = arith.mulf %in, %in : f32
          %24 = arith.addf %23, %out : f32
          linalg.yield %24 : f32
        } -> tensor<f32>
        %extracted = tensor.extract %18[] : tensor<f32>
        %19 = arith.divf %extracted, %cst_2 : f32
        %20 = arith.addf %19, %cst_3 : f32
        %21 = math.rsqrt %20 : f32
        %22 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%12, %extracted_slice_15 : tensor<768xf32>, tensor<768xf32>) outs(%9 : tensor<768xf32>) {
        ^bb0(%in: f32, %in_19: f32, %out: f32):
          %23 = arith.mulf %in, %21 : f32
          %24 = arith.mulf %23, %in_19 : f32
          linalg.yield %24 : f32
        } -> tensor<768xf32>
        cinm.yield %22 : tensor<768xf32>
      }
      %extracted_slice_16 = tensor.extract_slice %arg10[%arg16, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
      %extracted_slice_17 = tensor.extract_slice %arg12[%arg16, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
      %14:2 = cinm.compute_ -> tensor<2048xf32>, tensor<2048xf32>
           attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 6, 8>} {
        %16 = tensor.empty() : tensor<2048xf32>
        %17 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%16 : tensor<2048xf32>) {
        ^bb0(%out: f32):
          linalg.yield %cst : f32
        } -> tensor<2048xf32>
        %18 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_16, %13 : tensor<2048x768xf32>, tensor<768xf32>) outs(%17 : tensor<2048xf32>) {
        ^bb0(%in: f32, %in_19: f32, %out: f32):
          %20 = arith.mulf %in, %in_19 : f32
          %21 = arith.addf %out, %20 : f32
          linalg.yield %21 : f32
        } -> tensor<2048xf32>
        %19 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_17, %13 : tensor<2048x768xf32>, tensor<768xf32>) outs(%17 : tensor<2048xf32>) {
        ^bb0(%in: f32, %in_19: f32, %out: f32):
          %20 = arith.mulf %in, %in_19 : f32
          %21 = arith.addf %out, %20 : f32
          linalg.yield %21 : f32
        } -> tensor<2048xf32>
        cinm.yield %18, %19 : tensor<2048xf32>, tensor<2048xf32>
      }
      %mapped = linalg.map ins(%14#0, %14#1 : tensor<2048xf32>, tensor<2048xf32>) outs(%14#0 : tensor<2048xf32>)
        (%in: f32, %in_19: f32) {
          %16 = arith.negf %in : f32
          %17 = math.exp %16 : f32
          %18 = arith.addf %17, %cst_4 : f32
          %19 = arith.divf %cst_4, %18 : f32
          %20 = arith.mulf %in, %19 : f32
          %21 = arith.mulf %20, %in_19 : f32
          linalg.yield %21 : f32
        }
      %extracted_slice_18 = tensor.extract_slice %arg11[%arg16, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
      %15 = cinm.compute_ -> tensor<768xf32>
           attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 6, 8>} {
        %16 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_18, %mapped : tensor<768x2048xf32>, tensor<2048xf32>) outs(%13 : tensor<768xf32>) {
        ^bb0(%in: f32, %in_19: f32, %out: f32):
          %17 = arith.mulf %in, %in_19 : f32
          %18 = arith.addf %out, %17 : f32
          linalg.yield %18 : f32
        } -> tensor<768xf32>
        cinm.yield %16 : tensor<768xf32>
      }
      scf.yield %15, %inserted_slice, %4#2 : tensor<768xf32>, tensor<6x1024x768xf32>, tensor<6x1024x768xf32>
    }
    %1 = cinm.compute_ -> tensor<768xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %3 = tensor.empty() : tensor<f32>
      %4 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%3 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %5 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%0#0 : tensor<768xf32>) outs(%4 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %11 = arith.mulf %in, %in : f32
        %12 = arith.addf %11, %out : f32
        linalg.yield %12 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %5[] : tensor<f32>
      %6 = arith.divf %extracted, %cst_2 : f32
      %7 = arith.addf %6, %cst_3 : f32
      %8 = math.rsqrt %7 : f32
      %9 = tensor.empty() : tensor<768xf32>
      %10 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%0#0, %arg14 : tensor<768xf32>, tensor<768xf32>) outs(%9 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_7: f32, %out: f32):
        %11 = arith.mulf %in, %8 : f32
        %12 = arith.mulf %11, %in_7 : f32
        linalg.yield %12 : f32
      } -> tensor<768xf32>
      cinm.yield %10 : tensor<768xf32>
    }
    %2 = cinm.compute_ -> tensor<32000xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 8, 16>} {
      %3 = tensor.empty() : tensor<34048x768xf32>
      %4 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%3 : tensor<34048x768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<34048x768xf32>
      %inserted_slice = tensor.insert_slice %arg15 into %4[0, 0] [32000, 768] [1, 1] : tensor<32000x768xf32> into tensor<34048x768xf32>
      %5 = tensor.empty() : tensor<34048xf32>
      %6 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%5 : tensor<34048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<34048xf32>
      %7 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%inserted_slice, %1 : tensor<34048x768xf32>, tensor<768xf32>) outs(%6 : tensor<34048xf32>) {
      ^bb0(%in: f32, %in_8: f32, %out: f32):
        %8 = arith.mulf %in, %in_8 : f32
        %9 = arith.addf %out, %8 : f32
        linalg.yield %9 : f32
      } -> tensor<34048xf32>
      %extracted_slice_7 = tensor.extract_slice %7[0] [32000] [1] : tensor<34048xf32> to tensor<32000xf32>
      cinm.yield %extracted_slice_7 : tensor<32000xf32>
    }
    return %2 : tensor<32000xf32>
  }
  func.func @rot(%arg0: tensor<768xf32> {bufferization.writable = true}, %arg1: index, %arg2: f32, %arg3: f32) -> tensor<768xf32> {
    %c1 = arith.constant 1 : index
    %0 = arith.addi %arg1, %c1 : index
    %extracted = tensor.extract %arg0[%arg1] : tensor<768xf32>
    %extracted_0 = tensor.extract %arg0[%0] : tensor<768xf32>
    %1 = arith.mulf %extracted, %arg2 : f32
    %2 = arith.mulf %extracted_0, %arg3 : f32
    %3 = arith.subf %1, %2 : f32
    %inserted = tensor.insert %3 into %arg0[%arg1] : tensor<768xf32>
    %4 = arith.mulf %extracted, %arg3 : f32
    %5 = arith.mulf %extracted_0, %arg2 : f32
    %6 = arith.addf %4, %5 : f32
    %inserted_1 = tensor.insert %6 into %inserted[%0] : tensor<768xf32>
    %7 = bufferization.materialize_in_destination %inserted_1 in %arg0 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    return %7 : tensor<768xf32>
  }
  func.func @mha(%arg0: tensor<768xf32>, %arg1: tensor<1024x768xf32>, %arg2: tensor<1024x768xf32>, %arg3: index) -> tensor<768xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    %c48 = arith.constant 48 : index
    %c1024 = arith.constant 1024 : index
    %cst_0 = arith.constant 6.92820311 : f32
    %cst_1 = arith.constant 0xFF800000 : f32
    %0 = arith.addi %arg3, %c1 : index
    %1 = tensor.empty() : tensor<768xf32>
    %2 = scf.for %arg4 = %c0 to %c6 step %c1 iter_args(%arg5 = %1) -> (tensor<768xf32>) {
      %3 = arith.muli %arg4, %c48 : index
      %4 = tensor.empty() : tensor<1024xf32>
      %5 = scf.for %arg6 = %c0 to %0 step %c1 iter_args(%arg7 = %4) -> (tensor<1024xf32>) {
        %extracted_slice_2 = tensor.extract_slice %arg0[%3] [48] [1] : tensor<768xf32> to tensor<48xf32>
        %extracted_slice_3 = tensor.extract_slice %arg1[%arg6, %3] [1, 48] [1, 1] : tensor<1024x768xf32> to tensor<48xf32>
        %10 = cinm.compute_ -> f32
             attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 1, 8>} {
          %11 = tensor.empty() : tensor<f32>
          %12 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%11 : tensor<f32>) {
          ^bb0(%out: f32):
            linalg.yield %cst : f32
          } -> tensor<f32>
          %13 = linalg.generic {indexing_maps = [#map1, #map1, #map2], iterator_types = ["reduction"]} ins(%extracted_slice_2, %extracted_slice_3 : tensor<48xf32>, tensor<48xf32>) outs(%12 : tensor<f32>) {
          ^bb0(%in: f32, %in_4: f32, %out: f32):
            %15 = arith.mulf %in, %in_4 : f32
            %16 = arith.addf %15, %out : f32
            linalg.yield %16 : f32
          } -> tensor<f32>
          %extracted = tensor.extract %13[] : tensor<f32>
          %14 = arith.divf %extracted, %cst_0 : f32
          cinm.yield %14 : f32
        }
        %inserted = tensor.insert %10 into %arg7[%arg6] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      %6 = scf.for %arg6 = %0 to %c1024 step %c1 iter_args(%arg7 = %5) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_1 into %arg7[%arg6] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      %7 = cinm.compute_ -> tensor<1024xf32>
           attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
        %10 = tensor.empty() : tensor<f32>
        %11 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%10 : tensor<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_1 : f32
        } -> tensor<f32>
        %12 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%6 : tensor<1024xf32>) outs(%11 : tensor<f32>) {
        ^bb0(%in: f32, %out: f32):
          %17 = arith.maxnumf %in, %out : f32
          linalg.yield %17 : f32
        } -> tensor<f32>
        %extracted = tensor.extract %12[] : tensor<f32>
        %13 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%6 : tensor<1024xf32>) outs(%6 : tensor<1024xf32>) {
        ^bb0(%in: f32, %out: f32):
          %17 = arith.subf %in, %extracted : f32
          %18 = math.exp %17 : f32
          linalg.yield %18 : f32
        } -> tensor<1024xf32>
        %14 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%10 : tensor<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst : f32
        } -> tensor<f32>
        %15 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%13 : tensor<1024xf32>) outs(%14 : tensor<f32>) {
        ^bb0(%in: f32, %out: f32):
          %17 = arith.addf %in, %out : f32
          linalg.yield %17 : f32
        } -> tensor<f32>
        %extracted_2 = tensor.extract %15[] : tensor<f32>
        %16 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%13 : tensor<1024xf32>) outs(%6 : tensor<1024xf32>) {
        ^bb0(%in: f32, %out: f32):
          %17 = arith.divf %in, %extracted_2 : f32
          linalg.yield %17 : f32
        } -> tensor<1024xf32>
        cinm.yield %16 : tensor<1024xf32>
      }
      %extracted_slice = tensor.extract_slice %arg5[%3] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %8 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice : tensor<48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<48xf32>
      %inserted_slice = tensor.insert_slice %8 into %arg5[%3] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %9 = scf.for %arg6 = %c0 to %0 step %c1 iter_args(%arg7 = %inserted_slice) -> (tensor<768xf32>) {
        %extracted_slice_2 = tensor.extract_slice %arg7[%3] [48] [1] : tensor<768xf32> to tensor<48xf32>
        %extracted_slice_3 = tensor.extract_slice %arg2[%arg6, %3] [1, 48] [1, 1] : tensor<1024x768xf32> to tensor<48xf32>
        %extracted = tensor.extract %7[%arg6] : tensor<1024xf32>
        %10 = cinm.compute_ -> tensor<48xf32>
             attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 1, 8>} {
          %11 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%extracted_slice_2, %extracted_slice_3 : tensor<48xf32>, tensor<48xf32>) outs(%extracted_slice_2 : tensor<48xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %12 = arith.mulf %in_5, %extracted : f32
            %13 = arith.addf %in, %12 : f32
            linalg.yield %13 : f32
          } -> tensor<48xf32>
          cinm.yield %11 : tensor<48xf32>
        }
        %inserted_slice_4 = tensor.insert_slice %10 into %arg7[%3] [48] [1] : tensor<48xf32> into tensor<768xf32>
        scf.yield %inserted_slice_4 : tensor<768xf32>
      }
      scf.yield %9 : tensor<768xf32>
    }
    return %2 : tensor<768xf32>
  }
  func.func @rmsnorm(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>) -> tensor<768xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 9.99999974E-6 : f32
    %cst_1 = arith.constant 7.680000e+02 : f32
    %0 = cinm.compute_ -> tensor<768xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %1 = tensor.empty() : tensor<f32>
      %2 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %3 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%arg0 : tensor<768xf32>) outs(%2 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %9 = arith.mulf %in, %in : f32
        %10 = arith.addf %9, %out : f32
        linalg.yield %10 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %3[] : tensor<f32>
      %4 = arith.divf %extracted, %cst_1 : f32
      %5 = arith.addf %4, %cst_0 : f32
      %6 = math.rsqrt %5 : f32
      %7 = tensor.empty() : tensor<768xf32>
      %8 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<768xf32>, tensor<768xf32>) outs(%7 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_2: f32, %out: f32):
        %9 = arith.mulf %in, %6 : f32
        %10 = arith.mulf %9, %in_2 : f32
        linalg.yield %10 : f32
      } -> tensor<768xf32>
      cinm.yield %8 : tensor<768xf32>
    }
    return %0 : tensor<768xf32>
  }
  func.func @softmax(%arg0: tensor<1024xf32> {bufferization.writable = true}) -> tensor<1024xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %0 = cinm.compute_ -> tensor<1024xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %1 = tensor.empty() : tensor<f32>
      %2 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_0 : f32
      } -> tensor<f32>
      %3 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%arg0 : tensor<1024xf32>) outs(%2 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %8 = arith.maxnumf %in, %out : f32
        linalg.yield %8 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %3[] : tensor<f32>
      %4 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%arg0 : tensor<1024xf32>) outs(%arg0 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %8 = arith.subf %in, %extracted : f32
        %9 = math.exp %8 : f32
        linalg.yield %9 : f32
      } -> tensor<1024xf32>
      %5 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %6 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%4 : tensor<1024xf32>) outs(%5 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %8 = arith.addf %in, %out : f32
        linalg.yield %8 : f32
      } -> tensor<f32>
      %extracted_1 = tensor.extract %6[] : tensor<f32>
      %7 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%4 : tensor<1024xf32>) outs(%arg0 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %8 = arith.divf %in, %extracted_1 : f32
        linalg.yield %8 : f32
      } -> tensor<1024xf32>
      cinm.yield %7 : tensor<1024xf32>
    }
    return %0 : tensor<1024xf32>
  }
}

