#map = affine_map<() -> ()>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> ()>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d1)>
#map5 = affine_map<(d0, d1) -> (d0)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map7 = affine_map<(d0, d1, d2) -> ()>
#map8 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map9 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map10 = affine_map<(d0, d1) -> ()>
#upmem = #upmem.platform<type = v1A, dpus = 2560, tasklets = 24>
module {
  func.func @forward(%arg0: index, %arg1: index, %arg2: tensor<6x1024x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, bufferization.writable = true}, %arg3: tensor<6x1024x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, bufferization.writable = true}, %arg4: tensor<32000x768xf32> {bufferization.buffer_layout = #map3, cinm.static}, %arg5: tensor<6x768xf32> {bufferization.buffer_layout = #map3, cinm.static}, %arg6: tensor<6x768x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, cinm.static}, %arg7: tensor<6x768x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, cinm.static}, %arg8: tensor<6x768x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, cinm.static}, %arg9: tensor<6x768x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, cinm.static}, %arg10: tensor<6x2048x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, cinm.static}, %arg11: tensor<6x768x2048xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, cinm.static}, %arg12: tensor<6x2048x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, cinm.static}, %arg13: tensor<6x768xf32> {bufferization.buffer_layout = #map3, cinm.static}, %arg14: tensor<768xf32> {bufferization.buffer_layout = #map1, cinm.static}, %arg15: tensor<32000x768xf32> {bufferization.buffer_layout = #map3, cinm.static}) -> tensor<32000xf32> attributes {cinm.available_platforms = [#upmem]} {
    %cst = arith.constant 0xFFC00000 : f32
    %cst_0 = arith.constant 6.92820311 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 0xFF800000 : f32
    %c1024 = arith.constant 1024 : index
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
    %extracted_slice = tensor.extract_slice %arg4[%arg0, 0] [1, 768] [1, 1] : tensor<32000x768xf32> to tensor<768xf32>
    %extracted_slice_8 = tensor.extract_slice %arg5[0, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %0 = tensor.empty() : tensor<f32>
    %1 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<f32>
    %2 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%extracted_slice : tensor<768xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.mulf %in, %in : f32
      %917 = arith.addf %916, %out : f32
      linalg.yield %917 : f32
    } -> tensor<f32>
    %extracted = tensor.extract %2[] : tensor<f32>
    %3 = arith.divf %extracted, %cst_3 : f32
    %4 = arith.addf %3, %cst_4 : f32
    %5 = math.rsqrt %4 : f32
    %6 = tensor.empty() : tensor<768xf32>
    %7 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%extracted_slice, %5, %extracted_slice_8 : tensor<768xf32>, f32, tensor<768xf32>) outs(%6 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      linalg.yield %917 : f32
    } -> tensor<768xf32>
    %extracted_slice_9 = tensor.extract_slice %arg6[0, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_10 = tensor.extract_slice %arg7[0, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_11 = tensor.extract_slice %arg8[0, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %8 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%6 : tensor<768xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<768xf32>
    %9 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %7 : tensor<768x768xf32>, tensor<768xf32>) outs(%8 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<768xf32>
    %extracted_slice_12 = tensor.extract_slice %arg2[0, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %10 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_12 : tensor<768xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<768xf32>
    %11 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_10, %7 : tensor<768x768xf32>, tensor<768xf32>) outs(%10 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<768xf32>
    %extracted_slice_13 = tensor.extract_slice %arg3[0, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %12 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_13 : tensor<768xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<768xf32>
    %13 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_11, %7 : tensor<768x768xf32>, tensor<768xf32>) outs(%12 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<768xf32>
    %inserted_slice = tensor.insert_slice %13 into %arg3[0, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %14 = arith.index_cast %arg1 : index to i64
    %15 = arith.uitofp %14 : i64 to f32
    %16:2 = scf.for %arg16 = %c0 to %c768 step %c2 iter_args(%arg17 = %9, %arg18 = %11) -> (tensor<768xf32>, tensor<768xf32>) {
      %916 = arith.remui %arg16, %c48 : index
      %917 = arith.index_cast %916 : index to i64
      %918 = arith.uitofp %917 : i64 to f32
      %919 = arith.divf %918, %cst_6 : f32
      %920 = math.powf %cst_7, %919 : f32
      %921 = arith.divf %cst_5, %920 : f32
      %922 = arith.mulf %15, %921 : f32
      %923 = math.cos %922 : f32
      %924 = math.sin %922 : f32
      %925 = arith.addi %arg16, %c1 : index
      %extracted_1058 = tensor.extract %arg17[%arg16] : tensor<768xf32>
      %extracted_1059 = tensor.extract %arg17[%925] : tensor<768xf32>
      %926 = arith.mulf %extracted_1058, %923 : f32
      %927 = arith.mulf %extracted_1059, %924 : f32
      %928 = arith.subf %926, %927 : f32
      %inserted = tensor.insert %928 into %arg17[%arg16] : tensor<768xf32>
      %929 = arith.mulf %extracted_1058, %924 : f32
      %930 = arith.mulf %extracted_1059, %923 : f32
      %931 = arith.addf %929, %930 : f32
      %inserted_1060 = tensor.insert %931 into %inserted[%925] : tensor<768xf32>
      %932 = bufferization.materialize_in_destination %inserted_1060 in %arg17 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      %933 = arith.cmpi ult, %arg16, %c768 : index
      %934 = scf.if %933 -> (tensor<768xf32>) {
        %extracted_1061 = tensor.extract %arg18[%arg16] : tensor<768xf32>
        %extracted_1062 = tensor.extract %arg18[%925] : tensor<768xf32>
        %935 = arith.mulf %extracted_1061, %923 : f32
        %936 = arith.mulf %extracted_1062, %924 : f32
        %937 = arith.subf %935, %936 : f32
        %inserted_1063 = tensor.insert %937 into %arg18[%arg16] : tensor<768xf32>
        %938 = arith.mulf %extracted_1061, %924 : f32
        %939 = arith.mulf %extracted_1062, %923 : f32
        %940 = arith.addf %938, %939 : f32
        %inserted_1064 = tensor.insert %940 into %inserted_1063[%925] : tensor<768xf32>
        %941 = bufferization.materialize_in_destination %inserted_1064 in %arg18 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
        scf.yield %941 : tensor<768xf32>
      } else {
        scf.yield %arg18 : tensor<768xf32>
      }
      scf.yield %932, %934 : tensor<768xf32>, tensor<768xf32>
    }
    %inserted_slice_14 = tensor.insert_slice %16#1 into %arg2[0, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %extracted_slice_15 = tensor.extract_slice %inserted_slice_14[0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %extracted_slice_16 = tensor.extract_slice %inserted_slice[0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %17 = arith.addi %arg1, %c1 : index
    %extracted_slice_17 = tensor.extract_slice %16#0[0] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_18 = tensor.extract_slice %extracted_slice_15[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %18 = tensor.empty() : tensor<1024xf32>
    %19 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%18 : tensor<1024xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1024xf32>
    %20 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_18, %extracted_slice_17 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %21 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%20 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %22 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %21) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %23 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    } -> tensor<f32>
    %24 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%22 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_19 = tensor.extract %24[] : tensor<f32>
    %25 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%22, %extracted_19 : tensor<1024xf32>, f32) outs(%22 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %26 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%25 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_20 = tensor.extract %26[] : tensor<f32>
    %expanded = tensor.expand_shape %25 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_21 = tensor.extract_slice %extracted_slice_16[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_22 = tensor.extract_slice %6[0] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %from_elements = tensor.from_elements %c1, %c48 : tensor<2xindex>
    %reshape = tensor.reshape %extracted_slice_22(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %27 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %28 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded, %extracted_20, %extracted_slice_21 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%27 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed = tensor.collapse_shape %28 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_23 = tensor.insert_slice %collapsed into %7[0] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_24 = tensor.extract_slice %16#0[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_25 = tensor.extract_slice %extracted_slice_15[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %29 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_25, %extracted_slice_24 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %30 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%29 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %31 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %30) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %32 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%31 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_26 = tensor.extract %32[] : tensor<f32>
    %33 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%31, %extracted_26 : tensor<1024xf32>, f32) outs(%31 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %34 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%33 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_27 = tensor.extract %34[] : tensor<f32>
    %expanded_28 = tensor.expand_shape %33 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_29 = tensor.extract_slice %extracted_slice_16[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_30 = tensor.extract_slice %inserted_slice_23[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_31 = tensor.reshape %extracted_slice_30(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %35 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_31 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %36 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_28, %extracted_27, %extracted_slice_29 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%35 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_32 = tensor.collapse_shape %36 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_33 = tensor.insert_slice %collapsed_32 into %inserted_slice_23[48] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_34 = tensor.extract_slice %16#0[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_35 = tensor.extract_slice %extracted_slice_15[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %37 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_35, %extracted_slice_34 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %38 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%37 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %39 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %38) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %40 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%39 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_36 = tensor.extract %40[] : tensor<f32>
    %41 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%39, %extracted_36 : tensor<1024xf32>, f32) outs(%39 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %42 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%41 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_37 = tensor.extract %42[] : tensor<f32>
    %expanded_38 = tensor.expand_shape %41 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_39 = tensor.extract_slice %extracted_slice_16[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_40 = tensor.extract_slice %inserted_slice_33[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_41 = tensor.reshape %extracted_slice_40(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %43 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_41 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %44 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_38, %extracted_37, %extracted_slice_39 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%43 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_42 = tensor.collapse_shape %44 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_43 = tensor.insert_slice %collapsed_42 into %inserted_slice_33[96] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_44 = tensor.extract_slice %16#0[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_45 = tensor.extract_slice %extracted_slice_15[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %45 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_45, %extracted_slice_44 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %46 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%45 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %47 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %46) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %48 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%47 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_46 = tensor.extract %48[] : tensor<f32>
    %49 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%47, %extracted_46 : tensor<1024xf32>, f32) outs(%47 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %50 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%49 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_47 = tensor.extract %50[] : tensor<f32>
    %expanded_48 = tensor.expand_shape %49 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_49 = tensor.extract_slice %extracted_slice_16[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_50 = tensor.extract_slice %inserted_slice_43[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_51 = tensor.reshape %extracted_slice_50(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %51 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_51 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %52 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_48, %extracted_47, %extracted_slice_49 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%51 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_52 = tensor.collapse_shape %52 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_53 = tensor.insert_slice %collapsed_52 into %inserted_slice_43[144] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_54 = tensor.extract_slice %16#0[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_55 = tensor.extract_slice %extracted_slice_15[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %53 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_55, %extracted_slice_54 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %54 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%53 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %55 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %54) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %56 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%55 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_56 = tensor.extract %56[] : tensor<f32>
    %57 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%55, %extracted_56 : tensor<1024xf32>, f32) outs(%55 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %58 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%57 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_57 = tensor.extract %58[] : tensor<f32>
    %expanded_58 = tensor.expand_shape %57 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_59 = tensor.extract_slice %extracted_slice_16[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_60 = tensor.extract_slice %inserted_slice_53[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_61 = tensor.reshape %extracted_slice_60(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %59 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_61 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %60 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_58, %extracted_57, %extracted_slice_59 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%59 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_62 = tensor.collapse_shape %60 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_63 = tensor.insert_slice %collapsed_62 into %inserted_slice_53[192] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_64 = tensor.extract_slice %16#0[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_65 = tensor.extract_slice %extracted_slice_15[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %61 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_65, %extracted_slice_64 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %62 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%61 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %63 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %62) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %64 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%63 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_66 = tensor.extract %64[] : tensor<f32>
    %65 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%63, %extracted_66 : tensor<1024xf32>, f32) outs(%63 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %66 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%65 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_67 = tensor.extract %66[] : tensor<f32>
    %expanded_68 = tensor.expand_shape %65 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_69 = tensor.extract_slice %extracted_slice_16[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_70 = tensor.extract_slice %inserted_slice_63[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_71 = tensor.reshape %extracted_slice_70(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %67 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_71 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %68 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_68, %extracted_67, %extracted_slice_69 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%67 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_72 = tensor.collapse_shape %68 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_73 = tensor.insert_slice %collapsed_72 into %inserted_slice_63[240] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_74 = tensor.extract_slice %16#0[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_75 = tensor.extract_slice %extracted_slice_15[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %69 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_75, %extracted_slice_74 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %70 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%69 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %71 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %70) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %72 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%71 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_76 = tensor.extract %72[] : tensor<f32>
    %73 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%71, %extracted_76 : tensor<1024xf32>, f32) outs(%71 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %74 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%73 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_77 = tensor.extract %74[] : tensor<f32>
    %expanded_78 = tensor.expand_shape %73 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_79 = tensor.extract_slice %extracted_slice_16[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_80 = tensor.extract_slice %inserted_slice_73[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_81 = tensor.reshape %extracted_slice_80(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %75 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_81 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %76 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_78, %extracted_77, %extracted_slice_79 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%75 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_82 = tensor.collapse_shape %76 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_83 = tensor.insert_slice %collapsed_82 into %inserted_slice_73[288] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_84 = tensor.extract_slice %16#0[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_85 = tensor.extract_slice %extracted_slice_15[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %77 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_85, %extracted_slice_84 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %78 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%77 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %79 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %78) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %80 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%79 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_86 = tensor.extract %80[] : tensor<f32>
    %81 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%79, %extracted_86 : tensor<1024xf32>, f32) outs(%79 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %82 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%81 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_87 = tensor.extract %82[] : tensor<f32>
    %expanded_88 = tensor.expand_shape %81 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_89 = tensor.extract_slice %extracted_slice_16[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_90 = tensor.extract_slice %inserted_slice_83[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_91 = tensor.reshape %extracted_slice_90(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %83 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_91 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %84 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_88, %extracted_87, %extracted_slice_89 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%83 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_92 = tensor.collapse_shape %84 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_93 = tensor.insert_slice %collapsed_92 into %inserted_slice_83[336] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_94 = tensor.extract_slice %16#0[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_95 = tensor.extract_slice %extracted_slice_15[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %85 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_95, %extracted_slice_94 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %86 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%85 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %87 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %86) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %88 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%87 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_96 = tensor.extract %88[] : tensor<f32>
    %89 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%87, %extracted_96 : tensor<1024xf32>, f32) outs(%87 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %90 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%89 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_97 = tensor.extract %90[] : tensor<f32>
    %expanded_98 = tensor.expand_shape %89 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_99 = tensor.extract_slice %extracted_slice_16[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_100 = tensor.extract_slice %inserted_slice_93[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_101 = tensor.reshape %extracted_slice_100(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %91 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_101 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %92 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_98, %extracted_97, %extracted_slice_99 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%91 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_102 = tensor.collapse_shape %92 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_103 = tensor.insert_slice %collapsed_102 into %inserted_slice_93[384] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_104 = tensor.extract_slice %16#0[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_105 = tensor.extract_slice %extracted_slice_15[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %93 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_105, %extracted_slice_104 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %94 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%93 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %95 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %94) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %96 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%95 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_106 = tensor.extract %96[] : tensor<f32>
    %97 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%95, %extracted_106 : tensor<1024xf32>, f32) outs(%95 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %98 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%97 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_107 = tensor.extract %98[] : tensor<f32>
    %expanded_108 = tensor.expand_shape %97 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_109 = tensor.extract_slice %extracted_slice_16[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_110 = tensor.extract_slice %inserted_slice_103[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_111 = tensor.reshape %extracted_slice_110(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %99 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_111 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %100 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_108, %extracted_107, %extracted_slice_109 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%99 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_112 = tensor.collapse_shape %100 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_113 = tensor.insert_slice %collapsed_112 into %inserted_slice_103[432] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_114 = tensor.extract_slice %16#0[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_115 = tensor.extract_slice %extracted_slice_15[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %101 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_115, %extracted_slice_114 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %102 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%101 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %103 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %102) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %104 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%103 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_116 = tensor.extract %104[] : tensor<f32>
    %105 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%103, %extracted_116 : tensor<1024xf32>, f32) outs(%103 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %106 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%105 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_117 = tensor.extract %106[] : tensor<f32>
    %expanded_118 = tensor.expand_shape %105 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_119 = tensor.extract_slice %extracted_slice_16[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_120 = tensor.extract_slice %inserted_slice_113[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_121 = tensor.reshape %extracted_slice_120(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %107 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_121 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %108 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_118, %extracted_117, %extracted_slice_119 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%107 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_122 = tensor.collapse_shape %108 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_123 = tensor.insert_slice %collapsed_122 into %inserted_slice_113[480] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_124 = tensor.extract_slice %16#0[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_125 = tensor.extract_slice %extracted_slice_15[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %109 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_125, %extracted_slice_124 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %110 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%109 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %111 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %110) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %112 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%111 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_126 = tensor.extract %112[] : tensor<f32>
    %113 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%111, %extracted_126 : tensor<1024xf32>, f32) outs(%111 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %114 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%113 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_127 = tensor.extract %114[] : tensor<f32>
    %expanded_128 = tensor.expand_shape %113 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_129 = tensor.extract_slice %extracted_slice_16[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_130 = tensor.extract_slice %inserted_slice_123[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_131 = tensor.reshape %extracted_slice_130(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %115 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_131 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %116 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_128, %extracted_127, %extracted_slice_129 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%115 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_132 = tensor.collapse_shape %116 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_133 = tensor.insert_slice %collapsed_132 into %inserted_slice_123[528] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_134 = tensor.extract_slice %16#0[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_135 = tensor.extract_slice %extracted_slice_15[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %117 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_135, %extracted_slice_134 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %118 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%117 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %119 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %118) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %120 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%119 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_136 = tensor.extract %120[] : tensor<f32>
    %121 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%119, %extracted_136 : tensor<1024xf32>, f32) outs(%119 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %122 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%121 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_137 = tensor.extract %122[] : tensor<f32>
    %expanded_138 = tensor.expand_shape %121 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_139 = tensor.extract_slice %extracted_slice_16[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_140 = tensor.extract_slice %inserted_slice_133[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_141 = tensor.reshape %extracted_slice_140(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %123 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_141 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %124 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_138, %extracted_137, %extracted_slice_139 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%123 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_142 = tensor.collapse_shape %124 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_143 = tensor.insert_slice %collapsed_142 into %inserted_slice_133[576] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_144 = tensor.extract_slice %16#0[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_145 = tensor.extract_slice %extracted_slice_15[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %125 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_145, %extracted_slice_144 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %126 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%125 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %127 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %126) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %128 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%127 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_146 = tensor.extract %128[] : tensor<f32>
    %129 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%127, %extracted_146 : tensor<1024xf32>, f32) outs(%127 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %130 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%129 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_147 = tensor.extract %130[] : tensor<f32>
    %expanded_148 = tensor.expand_shape %129 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_149 = tensor.extract_slice %extracted_slice_16[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_150 = tensor.extract_slice %inserted_slice_143[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_151 = tensor.reshape %extracted_slice_150(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %131 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_151 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %132 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_148, %extracted_147, %extracted_slice_149 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%131 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_152 = tensor.collapse_shape %132 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_153 = tensor.insert_slice %collapsed_152 into %inserted_slice_143[624] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_154 = tensor.extract_slice %16#0[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_155 = tensor.extract_slice %extracted_slice_15[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %133 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_155, %extracted_slice_154 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %134 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%133 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %135 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %134) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %136 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%135 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_156 = tensor.extract %136[] : tensor<f32>
    %137 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%135, %extracted_156 : tensor<1024xf32>, f32) outs(%135 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %138 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%137 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_157 = tensor.extract %138[] : tensor<f32>
    %expanded_158 = tensor.expand_shape %137 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_159 = tensor.extract_slice %extracted_slice_16[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_160 = tensor.extract_slice %inserted_slice_153[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_161 = tensor.reshape %extracted_slice_160(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %139 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_161 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %140 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_158, %extracted_157, %extracted_slice_159 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%139 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_162 = tensor.collapse_shape %140 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_163 = tensor.insert_slice %collapsed_162 into %inserted_slice_153[672] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_164 = tensor.extract_slice %16#0[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_165 = tensor.extract_slice %extracted_slice_15[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %141 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_165, %extracted_slice_164 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %142 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%141 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %143 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %142) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %144 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%143 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_166 = tensor.extract %144[] : tensor<f32>
    %145 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%143, %extracted_166 : tensor<1024xf32>, f32) outs(%143 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %146 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%145 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_167 = tensor.extract %146[] : tensor<f32>
    %expanded_168 = tensor.expand_shape %145 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_169 = tensor.extract_slice %extracted_slice_16[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_170 = tensor.extract_slice %inserted_slice_163[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_171 = tensor.reshape %extracted_slice_170(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %147 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_171 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %148 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_168, %extracted_167, %extracted_slice_169 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%147 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_172 = tensor.collapse_shape %148 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_173 = tensor.insert_slice %collapsed_172 into %inserted_slice_163[720] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %149 = bufferization.materialize_in_destination %inserted_slice_173 in %7 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_174 = tensor.extract_slice %arg9[0, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %150 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_174, %149 : tensor<768x768xf32>, tensor<768xf32>) outs(%extracted_slice : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<768xf32>
    %extracted_slice_175 = tensor.extract_slice %arg13[0, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %151 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%150 : tensor<768xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.mulf %in, %in : f32
      %917 = arith.addf %916, %out : f32
      linalg.yield %917 : f32
    } -> tensor<f32>
    %extracted_176 = tensor.extract %151[] : tensor<f32>
    %152 = arith.divf %extracted_176, %cst_3 : f32
    %153 = arith.addf %152, %cst_4 : f32
    %154 = math.rsqrt %153 : f32
    %155 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%150, %154, %extracted_slice_175 : tensor<768xf32>, f32, tensor<768xf32>) outs(%149 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      linalg.yield %917 : f32
    } -> tensor<768xf32>
    %156 = bufferization.materialize_in_destination %155 in %149 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_177 = tensor.extract_slice %arg10[0, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %extracted_slice_178 = tensor.extract_slice %arg12[0, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %157 = tensor.empty() : tensor<2048xf32>
    %158 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%157 : tensor<2048xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<2048xf32>
    %159 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_177, %156 : tensor<2048x768xf32>, tensor<768xf32>) outs(%158 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<2048xf32>
    %160 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_178, %156 : tensor<2048x768xf32>, tensor<768xf32>) outs(%158 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<2048xf32>
    %extracted_slice_179 = tensor.extract_slice %arg11[0, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
    %161:2 = linalg.generic {indexing_maps = [#map3, #map4, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_179, %160 : tensor<768x2048xf32>, tensor<2048xf32>) outs(%159, %150 : tensor<2048xf32>, tensor<768xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32, %out_1059: f32):
      %916 = arith.negf %out : f32
      %917 = math.exp %916 : f32
      %918 = arith.addf %917, %cst_5 : f32
      %919 = arith.divf %cst_5, %918 : f32
      %920 = arith.mulf %out, %919 : f32
      %921 = arith.mulf %920, %in_1058 : f32
      %922 = arith.mulf %in, %921 : f32
      %923 = arith.addf %out_1059, %922 : f32
      linalg.yield %921, %923 : f32, f32
    } -> (tensor<2048xf32>, tensor<768xf32>)
    %extracted_slice_180 = tensor.extract_slice %arg5[1, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %162 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%161#1 : tensor<768xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.mulf %in, %in : f32
      %917 = arith.addf %916, %out : f32
      linalg.yield %917 : f32
    } -> tensor<f32>
    %extracted_181 = tensor.extract %162[] : tensor<f32>
    %163 = arith.divf %extracted_181, %cst_3 : f32
    %164 = arith.addf %163, %cst_4 : f32
    %165 = math.rsqrt %164 : f32
    %166 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%161#1, %165, %extracted_slice_180 : tensor<768xf32>, f32, tensor<768xf32>) outs(%6 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      linalg.yield %917 : f32
    } -> tensor<768xf32>
    %extracted_slice_182 = tensor.extract_slice %arg6[1, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_183 = tensor.extract_slice %arg7[1, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_184 = tensor.extract_slice %arg8[1, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %167 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_182, %166 : tensor<768x768xf32>, tensor<768xf32>) outs(%8 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<768xf32>
    %extracted_slice_185 = tensor.extract_slice %inserted_slice_14[1, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %168 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_185 : tensor<768xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<768xf32>
    %169 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_183, %166 : tensor<768x768xf32>, tensor<768xf32>) outs(%168 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<768xf32>
    %extracted_slice_186 = tensor.extract_slice %inserted_slice[1, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %170 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_186 : tensor<768xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<768xf32>
    %171 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_184, %166 : tensor<768x768xf32>, tensor<768xf32>) outs(%170 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<768xf32>
    %inserted_slice_187 = tensor.insert_slice %171 into %inserted_slice[1, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %172:2 = scf.for %arg16 = %c0 to %c768 step %c2 iter_args(%arg17 = %167, %arg18 = %169) -> (tensor<768xf32>, tensor<768xf32>) {
      %916 = arith.remui %arg16, %c48 : index
      %917 = arith.index_cast %916 : index to i64
      %918 = arith.uitofp %917 : i64 to f32
      %919 = arith.divf %918, %cst_6 : f32
      %920 = math.powf %cst_7, %919 : f32
      %921 = arith.divf %cst_5, %920 : f32
      %922 = arith.mulf %15, %921 : f32
      %923 = math.cos %922 : f32
      %924 = math.sin %922 : f32
      %925 = arith.addi %arg16, %c1 : index
      %extracted_1058 = tensor.extract %arg17[%arg16] : tensor<768xf32>
      %extracted_1059 = tensor.extract %arg17[%925] : tensor<768xf32>
      %926 = arith.mulf %extracted_1058, %923 : f32
      %927 = arith.mulf %extracted_1059, %924 : f32
      %928 = arith.subf %926, %927 : f32
      %inserted = tensor.insert %928 into %arg17[%arg16] : tensor<768xf32>
      %929 = arith.mulf %extracted_1058, %924 : f32
      %930 = arith.mulf %extracted_1059, %923 : f32
      %931 = arith.addf %929, %930 : f32
      %inserted_1060 = tensor.insert %931 into %inserted[%925] : tensor<768xf32>
      %932 = bufferization.materialize_in_destination %inserted_1060 in %arg17 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      %933 = arith.cmpi ult, %arg16, %c768 : index
      %934 = scf.if %933 -> (tensor<768xf32>) {
        %extracted_1061 = tensor.extract %arg18[%arg16] : tensor<768xf32>
        %extracted_1062 = tensor.extract %arg18[%925] : tensor<768xf32>
        %935 = arith.mulf %extracted_1061, %923 : f32
        %936 = arith.mulf %extracted_1062, %924 : f32
        %937 = arith.subf %935, %936 : f32
        %inserted_1063 = tensor.insert %937 into %arg18[%arg16] : tensor<768xf32>
        %938 = arith.mulf %extracted_1061, %924 : f32
        %939 = arith.mulf %extracted_1062, %923 : f32
        %940 = arith.addf %938, %939 : f32
        %inserted_1064 = tensor.insert %940 into %inserted_1063[%925] : tensor<768xf32>
        %941 = bufferization.materialize_in_destination %inserted_1064 in %arg18 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
        scf.yield %941 : tensor<768xf32>
      } else {
        scf.yield %arg18 : tensor<768xf32>
      }
      scf.yield %932, %934 : tensor<768xf32>, tensor<768xf32>
    }
    %inserted_slice_188 = tensor.insert_slice %172#1 into %inserted_slice_14[1, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %extracted_slice_189 = tensor.extract_slice %inserted_slice_188[1, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %extracted_slice_190 = tensor.extract_slice %inserted_slice_187[1, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %extracted_slice_191 = tensor.extract_slice %172#0[0] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_192 = tensor.extract_slice %extracted_slice_189[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %173 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_192, %extracted_slice_191 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %174 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%173 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %175 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %174) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %176 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%175 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_193 = tensor.extract %176[] : tensor<f32>
    %177 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%175, %extracted_193 : tensor<1024xf32>, f32) outs(%175 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %178 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%177 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_194 = tensor.extract %178[] : tensor<f32>
    %expanded_195 = tensor.expand_shape %177 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_196 = tensor.extract_slice %extracted_slice_190[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %179 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_195, %extracted_194, %extracted_slice_196 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%27 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_197 = tensor.collapse_shape %179 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_198 = tensor.insert_slice %collapsed_197 into %166[0] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_199 = tensor.extract_slice %172#0[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_200 = tensor.extract_slice %extracted_slice_189[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %180 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_200, %extracted_slice_199 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %181 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%180 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %182 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %181) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %183 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%182 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_201 = tensor.extract %183[] : tensor<f32>
    %184 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%182, %extracted_201 : tensor<1024xf32>, f32) outs(%182 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %185 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%184 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_202 = tensor.extract %185[] : tensor<f32>
    %expanded_203 = tensor.expand_shape %184 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_204 = tensor.extract_slice %extracted_slice_190[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_205 = tensor.extract_slice %inserted_slice_198[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_206 = tensor.reshape %extracted_slice_205(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %186 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_206 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %187 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_203, %extracted_202, %extracted_slice_204 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%186 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_207 = tensor.collapse_shape %187 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_208 = tensor.insert_slice %collapsed_207 into %inserted_slice_198[48] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_209 = tensor.extract_slice %172#0[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_210 = tensor.extract_slice %extracted_slice_189[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %188 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_210, %extracted_slice_209 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %189 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%188 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %190 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %189) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %191 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%190 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_211 = tensor.extract %191[] : tensor<f32>
    %192 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%190, %extracted_211 : tensor<1024xf32>, f32) outs(%190 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %193 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%192 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_212 = tensor.extract %193[] : tensor<f32>
    %expanded_213 = tensor.expand_shape %192 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_214 = tensor.extract_slice %extracted_slice_190[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_215 = tensor.extract_slice %inserted_slice_208[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_216 = tensor.reshape %extracted_slice_215(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %194 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_216 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %195 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_213, %extracted_212, %extracted_slice_214 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%194 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_217 = tensor.collapse_shape %195 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_218 = tensor.insert_slice %collapsed_217 into %inserted_slice_208[96] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_219 = tensor.extract_slice %172#0[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_220 = tensor.extract_slice %extracted_slice_189[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %196 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_220, %extracted_slice_219 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %197 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%196 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %198 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %197) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %199 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%198 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_221 = tensor.extract %199[] : tensor<f32>
    %200 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%198, %extracted_221 : tensor<1024xf32>, f32) outs(%198 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %201 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%200 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_222 = tensor.extract %201[] : tensor<f32>
    %expanded_223 = tensor.expand_shape %200 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_224 = tensor.extract_slice %extracted_slice_190[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_225 = tensor.extract_slice %inserted_slice_218[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_226 = tensor.reshape %extracted_slice_225(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %202 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_226 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %203 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_223, %extracted_222, %extracted_slice_224 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%202 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_227 = tensor.collapse_shape %203 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_228 = tensor.insert_slice %collapsed_227 into %inserted_slice_218[144] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_229 = tensor.extract_slice %172#0[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_230 = tensor.extract_slice %extracted_slice_189[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %204 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_230, %extracted_slice_229 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %205 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%204 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %206 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %205) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %207 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%206 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_231 = tensor.extract %207[] : tensor<f32>
    %208 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%206, %extracted_231 : tensor<1024xf32>, f32) outs(%206 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %209 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%208 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_232 = tensor.extract %209[] : tensor<f32>
    %expanded_233 = tensor.expand_shape %208 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_234 = tensor.extract_slice %extracted_slice_190[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_235 = tensor.extract_slice %inserted_slice_228[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_236 = tensor.reshape %extracted_slice_235(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %210 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_236 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %211 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_233, %extracted_232, %extracted_slice_234 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%210 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_237 = tensor.collapse_shape %211 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_238 = tensor.insert_slice %collapsed_237 into %inserted_slice_228[192] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_239 = tensor.extract_slice %172#0[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_240 = tensor.extract_slice %extracted_slice_189[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %212 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_240, %extracted_slice_239 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %213 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%212 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %214 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %213) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %215 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%214 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_241 = tensor.extract %215[] : tensor<f32>
    %216 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%214, %extracted_241 : tensor<1024xf32>, f32) outs(%214 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %217 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%216 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_242 = tensor.extract %217[] : tensor<f32>
    %expanded_243 = tensor.expand_shape %216 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_244 = tensor.extract_slice %extracted_slice_190[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_245 = tensor.extract_slice %inserted_slice_238[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_246 = tensor.reshape %extracted_slice_245(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %218 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_246 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %219 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_243, %extracted_242, %extracted_slice_244 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%218 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_247 = tensor.collapse_shape %219 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_248 = tensor.insert_slice %collapsed_247 into %inserted_slice_238[240] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_249 = tensor.extract_slice %172#0[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_250 = tensor.extract_slice %extracted_slice_189[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %220 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_250, %extracted_slice_249 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %221 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%220 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %222 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %221) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %223 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%222 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_251 = tensor.extract %223[] : tensor<f32>
    %224 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%222, %extracted_251 : tensor<1024xf32>, f32) outs(%222 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %225 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%224 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_252 = tensor.extract %225[] : tensor<f32>
    %expanded_253 = tensor.expand_shape %224 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_254 = tensor.extract_slice %extracted_slice_190[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_255 = tensor.extract_slice %inserted_slice_248[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_256 = tensor.reshape %extracted_slice_255(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %226 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_256 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %227 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_253, %extracted_252, %extracted_slice_254 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%226 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_257 = tensor.collapse_shape %227 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_258 = tensor.insert_slice %collapsed_257 into %inserted_slice_248[288] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_259 = tensor.extract_slice %172#0[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_260 = tensor.extract_slice %extracted_slice_189[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %228 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_260, %extracted_slice_259 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %229 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%228 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %230 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %229) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %231 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%230 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_261 = tensor.extract %231[] : tensor<f32>
    %232 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%230, %extracted_261 : tensor<1024xf32>, f32) outs(%230 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %233 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%232 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_262 = tensor.extract %233[] : tensor<f32>
    %expanded_263 = tensor.expand_shape %232 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_264 = tensor.extract_slice %extracted_slice_190[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_265 = tensor.extract_slice %inserted_slice_258[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_266 = tensor.reshape %extracted_slice_265(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %234 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_266 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %235 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_263, %extracted_262, %extracted_slice_264 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%234 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_267 = tensor.collapse_shape %235 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_268 = tensor.insert_slice %collapsed_267 into %inserted_slice_258[336] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_269 = tensor.extract_slice %172#0[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_270 = tensor.extract_slice %extracted_slice_189[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %236 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_270, %extracted_slice_269 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %237 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%236 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %238 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %237) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %239 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%238 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_271 = tensor.extract %239[] : tensor<f32>
    %240 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%238, %extracted_271 : tensor<1024xf32>, f32) outs(%238 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %241 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%240 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_272 = tensor.extract %241[] : tensor<f32>
    %expanded_273 = tensor.expand_shape %240 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_274 = tensor.extract_slice %extracted_slice_190[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_275 = tensor.extract_slice %inserted_slice_268[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_276 = tensor.reshape %extracted_slice_275(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %242 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_276 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %243 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_273, %extracted_272, %extracted_slice_274 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%242 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_277 = tensor.collapse_shape %243 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_278 = tensor.insert_slice %collapsed_277 into %inserted_slice_268[384] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_279 = tensor.extract_slice %172#0[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_280 = tensor.extract_slice %extracted_slice_189[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %244 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_280, %extracted_slice_279 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %245 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%244 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %246 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %245) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %247 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%246 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_281 = tensor.extract %247[] : tensor<f32>
    %248 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%246, %extracted_281 : tensor<1024xf32>, f32) outs(%246 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %249 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%248 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_282 = tensor.extract %249[] : tensor<f32>
    %expanded_283 = tensor.expand_shape %248 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_284 = tensor.extract_slice %extracted_slice_190[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_285 = tensor.extract_slice %inserted_slice_278[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_286 = tensor.reshape %extracted_slice_285(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %250 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_286 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %251 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_283, %extracted_282, %extracted_slice_284 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%250 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_287 = tensor.collapse_shape %251 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_288 = tensor.insert_slice %collapsed_287 into %inserted_slice_278[432] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_289 = tensor.extract_slice %172#0[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_290 = tensor.extract_slice %extracted_slice_189[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %252 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_290, %extracted_slice_289 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %253 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%252 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %254 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %253) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %255 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%254 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_291 = tensor.extract %255[] : tensor<f32>
    %256 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%254, %extracted_291 : tensor<1024xf32>, f32) outs(%254 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %257 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%256 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_292 = tensor.extract %257[] : tensor<f32>
    %expanded_293 = tensor.expand_shape %256 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_294 = tensor.extract_slice %extracted_slice_190[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_295 = tensor.extract_slice %inserted_slice_288[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_296 = tensor.reshape %extracted_slice_295(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %258 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_296 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %259 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_293, %extracted_292, %extracted_slice_294 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%258 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_297 = tensor.collapse_shape %259 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_298 = tensor.insert_slice %collapsed_297 into %inserted_slice_288[480] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_299 = tensor.extract_slice %172#0[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_300 = tensor.extract_slice %extracted_slice_189[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %260 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_300, %extracted_slice_299 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %261 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%260 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %262 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %261) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %263 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%262 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_301 = tensor.extract %263[] : tensor<f32>
    %264 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%262, %extracted_301 : tensor<1024xf32>, f32) outs(%262 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %265 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%264 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_302 = tensor.extract %265[] : tensor<f32>
    %expanded_303 = tensor.expand_shape %264 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_304 = tensor.extract_slice %extracted_slice_190[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_305 = tensor.extract_slice %inserted_slice_298[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_306 = tensor.reshape %extracted_slice_305(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %266 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_306 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %267 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_303, %extracted_302, %extracted_slice_304 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%266 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_307 = tensor.collapse_shape %267 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_308 = tensor.insert_slice %collapsed_307 into %inserted_slice_298[528] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_309 = tensor.extract_slice %172#0[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_310 = tensor.extract_slice %extracted_slice_189[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %268 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_310, %extracted_slice_309 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %269 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%268 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %270 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %269) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %271 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%270 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_311 = tensor.extract %271[] : tensor<f32>
    %272 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%270, %extracted_311 : tensor<1024xf32>, f32) outs(%270 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %273 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%272 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_312 = tensor.extract %273[] : tensor<f32>
    %expanded_313 = tensor.expand_shape %272 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_314 = tensor.extract_slice %extracted_slice_190[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_315 = tensor.extract_slice %inserted_slice_308[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_316 = tensor.reshape %extracted_slice_315(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %274 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_316 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %275 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_313, %extracted_312, %extracted_slice_314 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%274 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_317 = tensor.collapse_shape %275 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_318 = tensor.insert_slice %collapsed_317 into %inserted_slice_308[576] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_319 = tensor.extract_slice %172#0[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_320 = tensor.extract_slice %extracted_slice_189[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %276 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_320, %extracted_slice_319 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %277 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%276 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %278 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %277) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %279 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%278 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_321 = tensor.extract %279[] : tensor<f32>
    %280 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%278, %extracted_321 : tensor<1024xf32>, f32) outs(%278 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %281 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%280 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_322 = tensor.extract %281[] : tensor<f32>
    %expanded_323 = tensor.expand_shape %280 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_324 = tensor.extract_slice %extracted_slice_190[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_325 = tensor.extract_slice %inserted_slice_318[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_326 = tensor.reshape %extracted_slice_325(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %282 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_326 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %283 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_323, %extracted_322, %extracted_slice_324 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%282 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_327 = tensor.collapse_shape %283 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_328 = tensor.insert_slice %collapsed_327 into %inserted_slice_318[624] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_329 = tensor.extract_slice %172#0[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_330 = tensor.extract_slice %extracted_slice_189[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %284 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_330, %extracted_slice_329 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %285 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%284 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %286 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %285) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %287 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%286 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_331 = tensor.extract %287[] : tensor<f32>
    %288 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%286, %extracted_331 : tensor<1024xf32>, f32) outs(%286 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %289 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%288 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_332 = tensor.extract %289[] : tensor<f32>
    %expanded_333 = tensor.expand_shape %288 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_334 = tensor.extract_slice %extracted_slice_190[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_335 = tensor.extract_slice %inserted_slice_328[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_336 = tensor.reshape %extracted_slice_335(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %290 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_336 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %291 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_333, %extracted_332, %extracted_slice_334 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%290 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_337 = tensor.collapse_shape %291 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_338 = tensor.insert_slice %collapsed_337 into %inserted_slice_328[672] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_339 = tensor.extract_slice %172#0[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_340 = tensor.extract_slice %extracted_slice_189[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %292 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_340, %extracted_slice_339 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %293 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%292 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %294 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %293) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %295 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%294 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_341 = tensor.extract %295[] : tensor<f32>
    %296 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%294, %extracted_341 : tensor<1024xf32>, f32) outs(%294 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %297 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%296 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_342 = tensor.extract %297[] : tensor<f32>
    %expanded_343 = tensor.expand_shape %296 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_344 = tensor.extract_slice %extracted_slice_190[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_345 = tensor.extract_slice %inserted_slice_338[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_346 = tensor.reshape %extracted_slice_345(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %298 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_346 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %299 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_343, %extracted_342, %extracted_slice_344 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%298 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_347 = tensor.collapse_shape %299 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_348 = tensor.insert_slice %collapsed_347 into %inserted_slice_338[720] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %300 = bufferization.materialize_in_destination %inserted_slice_348 in %166 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_349 = tensor.extract_slice %arg9[1, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %301 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_349, %300 : tensor<768x768xf32>, tensor<768xf32>) outs(%161#1 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<768xf32>
    %extracted_slice_350 = tensor.extract_slice %arg13[1, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %302 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%301 : tensor<768xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.mulf %in, %in : f32
      %917 = arith.addf %916, %out : f32
      linalg.yield %917 : f32
    } -> tensor<f32>
    %extracted_351 = tensor.extract %302[] : tensor<f32>
    %303 = arith.divf %extracted_351, %cst_3 : f32
    %304 = arith.addf %303, %cst_4 : f32
    %305 = math.rsqrt %304 : f32
    %306 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%301, %305, %extracted_slice_350 : tensor<768xf32>, f32, tensor<768xf32>) outs(%300 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      linalg.yield %917 : f32
    } -> tensor<768xf32>
    %307 = bufferization.materialize_in_destination %306 in %300 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_352 = tensor.extract_slice %arg10[1, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %extracted_slice_353 = tensor.extract_slice %arg12[1, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %308 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_352, %307 : tensor<2048x768xf32>, tensor<768xf32>) outs(%158 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<2048xf32>
    %309 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_353, %307 : tensor<2048x768xf32>, tensor<768xf32>) outs(%158 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<2048xf32>
    %extracted_slice_354 = tensor.extract_slice %arg11[1, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
    %310:2 = linalg.generic {indexing_maps = [#map3, #map4, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_354, %309 : tensor<768x2048xf32>, tensor<2048xf32>) outs(%308, %301 : tensor<2048xf32>, tensor<768xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32, %out_1059: f32):
      %916 = arith.negf %out : f32
      %917 = math.exp %916 : f32
      %918 = arith.addf %917, %cst_5 : f32
      %919 = arith.divf %cst_5, %918 : f32
      %920 = arith.mulf %out, %919 : f32
      %921 = arith.mulf %920, %in_1058 : f32
      %922 = arith.mulf %in, %921 : f32
      %923 = arith.addf %out_1059, %922 : f32
      linalg.yield %921, %923 : f32, f32
    } -> (tensor<2048xf32>, tensor<768xf32>)
    %extracted_slice_355 = tensor.extract_slice %arg5[2, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %311 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%310#1 : tensor<768xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.mulf %in, %in : f32
      %917 = arith.addf %916, %out : f32
      linalg.yield %917 : f32
    } -> tensor<f32>
    %extracted_356 = tensor.extract %311[] : tensor<f32>
    %312 = arith.divf %extracted_356, %cst_3 : f32
    %313 = arith.addf %312, %cst_4 : f32
    %314 = math.rsqrt %313 : f32
    %315 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%310#1, %314, %extracted_slice_355 : tensor<768xf32>, f32, tensor<768xf32>) outs(%6 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      linalg.yield %917 : f32
    } -> tensor<768xf32>
    %extracted_slice_357 = tensor.extract_slice %arg6[2, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_358 = tensor.extract_slice %arg7[2, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_359 = tensor.extract_slice %arg8[2, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %316 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_357, %315 : tensor<768x768xf32>, tensor<768xf32>) outs(%8 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<768xf32>
    %extracted_slice_360 = tensor.extract_slice %inserted_slice_188[2, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %317 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_360 : tensor<768xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<768xf32>
    %318 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_358, %315 : tensor<768x768xf32>, tensor<768xf32>) outs(%317 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<768xf32>
    %extracted_slice_361 = tensor.extract_slice %inserted_slice_187[2, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %319 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_361 : tensor<768xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<768xf32>
    %320 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_359, %315 : tensor<768x768xf32>, tensor<768xf32>) outs(%319 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<768xf32>
    %inserted_slice_362 = tensor.insert_slice %320 into %inserted_slice_187[2, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %321:2 = scf.for %arg16 = %c0 to %c768 step %c2 iter_args(%arg17 = %316, %arg18 = %318) -> (tensor<768xf32>, tensor<768xf32>) {
      %916 = arith.remui %arg16, %c48 : index
      %917 = arith.index_cast %916 : index to i64
      %918 = arith.uitofp %917 : i64 to f32
      %919 = arith.divf %918, %cst_6 : f32
      %920 = math.powf %cst_7, %919 : f32
      %921 = arith.divf %cst_5, %920 : f32
      %922 = arith.mulf %15, %921 : f32
      %923 = math.cos %922 : f32
      %924 = math.sin %922 : f32
      %925 = arith.addi %arg16, %c1 : index
      %extracted_1058 = tensor.extract %arg17[%arg16] : tensor<768xf32>
      %extracted_1059 = tensor.extract %arg17[%925] : tensor<768xf32>
      %926 = arith.mulf %extracted_1058, %923 : f32
      %927 = arith.mulf %extracted_1059, %924 : f32
      %928 = arith.subf %926, %927 : f32
      %inserted = tensor.insert %928 into %arg17[%arg16] : tensor<768xf32>
      %929 = arith.mulf %extracted_1058, %924 : f32
      %930 = arith.mulf %extracted_1059, %923 : f32
      %931 = arith.addf %929, %930 : f32
      %inserted_1060 = tensor.insert %931 into %inserted[%925] : tensor<768xf32>
      %932 = bufferization.materialize_in_destination %inserted_1060 in %arg17 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      %933 = arith.cmpi ult, %arg16, %c768 : index
      %934 = scf.if %933 -> (tensor<768xf32>) {
        %extracted_1061 = tensor.extract %arg18[%arg16] : tensor<768xf32>
        %extracted_1062 = tensor.extract %arg18[%925] : tensor<768xf32>
        %935 = arith.mulf %extracted_1061, %923 : f32
        %936 = arith.mulf %extracted_1062, %924 : f32
        %937 = arith.subf %935, %936 : f32
        %inserted_1063 = tensor.insert %937 into %arg18[%arg16] : tensor<768xf32>
        %938 = arith.mulf %extracted_1061, %924 : f32
        %939 = arith.mulf %extracted_1062, %923 : f32
        %940 = arith.addf %938, %939 : f32
        %inserted_1064 = tensor.insert %940 into %inserted_1063[%925] : tensor<768xf32>
        %941 = bufferization.materialize_in_destination %inserted_1064 in %arg18 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
        scf.yield %941 : tensor<768xf32>
      } else {
        scf.yield %arg18 : tensor<768xf32>
      }
      scf.yield %932, %934 : tensor<768xf32>, tensor<768xf32>
    }
    %inserted_slice_363 = tensor.insert_slice %321#1 into %inserted_slice_188[2, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %extracted_slice_364 = tensor.extract_slice %inserted_slice_363[2, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %extracted_slice_365 = tensor.extract_slice %inserted_slice_362[2, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %extracted_slice_366 = tensor.extract_slice %321#0[0] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_367 = tensor.extract_slice %extracted_slice_364[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %322 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_367, %extracted_slice_366 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %323 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%322 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %324 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %323) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %325 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%324 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_368 = tensor.extract %325[] : tensor<f32>
    %326 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%324, %extracted_368 : tensor<1024xf32>, f32) outs(%324 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %327 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%326 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_369 = tensor.extract %327[] : tensor<f32>
    %expanded_370 = tensor.expand_shape %326 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_371 = tensor.extract_slice %extracted_slice_365[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %328 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_370, %extracted_369, %extracted_slice_371 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%27 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_372 = tensor.collapse_shape %328 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_373 = tensor.insert_slice %collapsed_372 into %315[0] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_374 = tensor.extract_slice %321#0[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_375 = tensor.extract_slice %extracted_slice_364[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %329 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_375, %extracted_slice_374 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %330 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%329 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %331 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %330) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %332 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%331 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_376 = tensor.extract %332[] : tensor<f32>
    %333 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%331, %extracted_376 : tensor<1024xf32>, f32) outs(%331 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %334 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%333 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_377 = tensor.extract %334[] : tensor<f32>
    %expanded_378 = tensor.expand_shape %333 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_379 = tensor.extract_slice %extracted_slice_365[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_380 = tensor.extract_slice %inserted_slice_373[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_381 = tensor.reshape %extracted_slice_380(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %335 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_381 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %336 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_378, %extracted_377, %extracted_slice_379 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%335 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_382 = tensor.collapse_shape %336 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_383 = tensor.insert_slice %collapsed_382 into %inserted_slice_373[48] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_384 = tensor.extract_slice %321#0[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_385 = tensor.extract_slice %extracted_slice_364[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %337 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_385, %extracted_slice_384 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %338 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%337 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %339 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %338) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %340 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%339 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_386 = tensor.extract %340[] : tensor<f32>
    %341 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%339, %extracted_386 : tensor<1024xf32>, f32) outs(%339 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %342 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%341 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_387 = tensor.extract %342[] : tensor<f32>
    %expanded_388 = tensor.expand_shape %341 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_389 = tensor.extract_slice %extracted_slice_365[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_390 = tensor.extract_slice %inserted_slice_383[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_391 = tensor.reshape %extracted_slice_390(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %343 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_391 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %344 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_388, %extracted_387, %extracted_slice_389 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%343 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_392 = tensor.collapse_shape %344 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_393 = tensor.insert_slice %collapsed_392 into %inserted_slice_383[96] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_394 = tensor.extract_slice %321#0[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_395 = tensor.extract_slice %extracted_slice_364[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %345 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_395, %extracted_slice_394 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %346 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%345 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %347 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %346) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %348 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%347 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_396 = tensor.extract %348[] : tensor<f32>
    %349 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%347, %extracted_396 : tensor<1024xf32>, f32) outs(%347 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %350 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%349 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_397 = tensor.extract %350[] : tensor<f32>
    %expanded_398 = tensor.expand_shape %349 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_399 = tensor.extract_slice %extracted_slice_365[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_400 = tensor.extract_slice %inserted_slice_393[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_401 = tensor.reshape %extracted_slice_400(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %351 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_401 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %352 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_398, %extracted_397, %extracted_slice_399 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%351 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_402 = tensor.collapse_shape %352 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_403 = tensor.insert_slice %collapsed_402 into %inserted_slice_393[144] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_404 = tensor.extract_slice %321#0[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_405 = tensor.extract_slice %extracted_slice_364[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %353 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_405, %extracted_slice_404 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %354 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%353 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %355 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %354) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %356 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%355 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_406 = tensor.extract %356[] : tensor<f32>
    %357 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%355, %extracted_406 : tensor<1024xf32>, f32) outs(%355 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %358 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%357 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_407 = tensor.extract %358[] : tensor<f32>
    %expanded_408 = tensor.expand_shape %357 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_409 = tensor.extract_slice %extracted_slice_365[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_410 = tensor.extract_slice %inserted_slice_403[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_411 = tensor.reshape %extracted_slice_410(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %359 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_411 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %360 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_408, %extracted_407, %extracted_slice_409 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%359 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_412 = tensor.collapse_shape %360 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_413 = tensor.insert_slice %collapsed_412 into %inserted_slice_403[192] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_414 = tensor.extract_slice %321#0[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_415 = tensor.extract_slice %extracted_slice_364[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %361 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_415, %extracted_slice_414 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %362 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%361 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %363 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %362) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %364 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%363 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_416 = tensor.extract %364[] : tensor<f32>
    %365 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%363, %extracted_416 : tensor<1024xf32>, f32) outs(%363 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %366 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%365 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_417 = tensor.extract %366[] : tensor<f32>
    %expanded_418 = tensor.expand_shape %365 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_419 = tensor.extract_slice %extracted_slice_365[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_420 = tensor.extract_slice %inserted_slice_413[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_421 = tensor.reshape %extracted_slice_420(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %367 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_421 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %368 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_418, %extracted_417, %extracted_slice_419 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%367 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_422 = tensor.collapse_shape %368 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_423 = tensor.insert_slice %collapsed_422 into %inserted_slice_413[240] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_424 = tensor.extract_slice %321#0[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_425 = tensor.extract_slice %extracted_slice_364[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %369 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_425, %extracted_slice_424 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %370 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%369 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %371 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %370) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %372 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%371 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_426 = tensor.extract %372[] : tensor<f32>
    %373 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%371, %extracted_426 : tensor<1024xf32>, f32) outs(%371 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %374 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%373 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_427 = tensor.extract %374[] : tensor<f32>
    %expanded_428 = tensor.expand_shape %373 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_429 = tensor.extract_slice %extracted_slice_365[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_430 = tensor.extract_slice %inserted_slice_423[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_431 = tensor.reshape %extracted_slice_430(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %375 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_431 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %376 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_428, %extracted_427, %extracted_slice_429 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%375 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_432 = tensor.collapse_shape %376 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_433 = tensor.insert_slice %collapsed_432 into %inserted_slice_423[288] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_434 = tensor.extract_slice %321#0[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_435 = tensor.extract_slice %extracted_slice_364[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %377 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_435, %extracted_slice_434 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %378 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%377 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %379 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %378) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %380 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%379 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_436 = tensor.extract %380[] : tensor<f32>
    %381 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%379, %extracted_436 : tensor<1024xf32>, f32) outs(%379 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %382 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%381 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_437 = tensor.extract %382[] : tensor<f32>
    %expanded_438 = tensor.expand_shape %381 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_439 = tensor.extract_slice %extracted_slice_365[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_440 = tensor.extract_slice %inserted_slice_433[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_441 = tensor.reshape %extracted_slice_440(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %383 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_441 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %384 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_438, %extracted_437, %extracted_slice_439 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%383 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_442 = tensor.collapse_shape %384 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_443 = tensor.insert_slice %collapsed_442 into %inserted_slice_433[336] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_444 = tensor.extract_slice %321#0[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_445 = tensor.extract_slice %extracted_slice_364[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %385 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_445, %extracted_slice_444 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %386 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%385 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %387 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %386) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %388 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%387 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_446 = tensor.extract %388[] : tensor<f32>
    %389 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%387, %extracted_446 : tensor<1024xf32>, f32) outs(%387 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %390 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%389 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_447 = tensor.extract %390[] : tensor<f32>
    %expanded_448 = tensor.expand_shape %389 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_449 = tensor.extract_slice %extracted_slice_365[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_450 = tensor.extract_slice %inserted_slice_443[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_451 = tensor.reshape %extracted_slice_450(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %391 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_451 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %392 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_448, %extracted_447, %extracted_slice_449 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%391 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_452 = tensor.collapse_shape %392 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_453 = tensor.insert_slice %collapsed_452 into %inserted_slice_443[384] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_454 = tensor.extract_slice %321#0[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_455 = tensor.extract_slice %extracted_slice_364[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %393 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_455, %extracted_slice_454 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %394 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%393 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %395 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %394) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %396 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%395 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_456 = tensor.extract %396[] : tensor<f32>
    %397 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%395, %extracted_456 : tensor<1024xf32>, f32) outs(%395 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %398 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%397 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_457 = tensor.extract %398[] : tensor<f32>
    %expanded_458 = tensor.expand_shape %397 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_459 = tensor.extract_slice %extracted_slice_365[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_460 = tensor.extract_slice %inserted_slice_453[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_461 = tensor.reshape %extracted_slice_460(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %399 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_461 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %400 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_458, %extracted_457, %extracted_slice_459 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%399 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_462 = tensor.collapse_shape %400 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_463 = tensor.insert_slice %collapsed_462 into %inserted_slice_453[432] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_464 = tensor.extract_slice %321#0[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_465 = tensor.extract_slice %extracted_slice_364[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %401 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_465, %extracted_slice_464 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %402 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%401 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %403 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %402) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %404 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%403 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_466 = tensor.extract %404[] : tensor<f32>
    %405 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%403, %extracted_466 : tensor<1024xf32>, f32) outs(%403 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %406 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%405 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_467 = tensor.extract %406[] : tensor<f32>
    %expanded_468 = tensor.expand_shape %405 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_469 = tensor.extract_slice %extracted_slice_365[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_470 = tensor.extract_slice %inserted_slice_463[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_471 = tensor.reshape %extracted_slice_470(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %407 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_471 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %408 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_468, %extracted_467, %extracted_slice_469 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%407 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_472 = tensor.collapse_shape %408 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_473 = tensor.insert_slice %collapsed_472 into %inserted_slice_463[480] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_474 = tensor.extract_slice %321#0[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_475 = tensor.extract_slice %extracted_slice_364[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %409 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_475, %extracted_slice_474 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %410 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%409 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %411 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %410) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %412 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%411 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_476 = tensor.extract %412[] : tensor<f32>
    %413 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%411, %extracted_476 : tensor<1024xf32>, f32) outs(%411 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %414 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%413 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_477 = tensor.extract %414[] : tensor<f32>
    %expanded_478 = tensor.expand_shape %413 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_479 = tensor.extract_slice %extracted_slice_365[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_480 = tensor.extract_slice %inserted_slice_473[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_481 = tensor.reshape %extracted_slice_480(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %415 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_481 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %416 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_478, %extracted_477, %extracted_slice_479 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%415 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_482 = tensor.collapse_shape %416 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_483 = tensor.insert_slice %collapsed_482 into %inserted_slice_473[528] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_484 = tensor.extract_slice %321#0[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_485 = tensor.extract_slice %extracted_slice_364[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %417 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_485, %extracted_slice_484 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %418 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%417 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %419 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %418) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %420 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%419 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_486 = tensor.extract %420[] : tensor<f32>
    %421 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%419, %extracted_486 : tensor<1024xf32>, f32) outs(%419 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %422 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%421 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_487 = tensor.extract %422[] : tensor<f32>
    %expanded_488 = tensor.expand_shape %421 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_489 = tensor.extract_slice %extracted_slice_365[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_490 = tensor.extract_slice %inserted_slice_483[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_491 = tensor.reshape %extracted_slice_490(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %423 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_491 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %424 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_488, %extracted_487, %extracted_slice_489 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%423 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_492 = tensor.collapse_shape %424 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_493 = tensor.insert_slice %collapsed_492 into %inserted_slice_483[576] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_494 = tensor.extract_slice %321#0[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_495 = tensor.extract_slice %extracted_slice_364[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %425 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_495, %extracted_slice_494 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %426 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%425 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %427 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %426) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %428 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%427 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_496 = tensor.extract %428[] : tensor<f32>
    %429 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%427, %extracted_496 : tensor<1024xf32>, f32) outs(%427 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %430 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%429 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_497 = tensor.extract %430[] : tensor<f32>
    %expanded_498 = tensor.expand_shape %429 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_499 = tensor.extract_slice %extracted_slice_365[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_500 = tensor.extract_slice %inserted_slice_493[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_501 = tensor.reshape %extracted_slice_500(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %431 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_501 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %432 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_498, %extracted_497, %extracted_slice_499 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%431 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_502 = tensor.collapse_shape %432 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_503 = tensor.insert_slice %collapsed_502 into %inserted_slice_493[624] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_504 = tensor.extract_slice %321#0[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_505 = tensor.extract_slice %extracted_slice_364[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %433 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_505, %extracted_slice_504 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %434 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%433 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %435 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %434) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %436 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%435 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_506 = tensor.extract %436[] : tensor<f32>
    %437 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%435, %extracted_506 : tensor<1024xf32>, f32) outs(%435 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %438 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%437 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_507 = tensor.extract %438[] : tensor<f32>
    %expanded_508 = tensor.expand_shape %437 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_509 = tensor.extract_slice %extracted_slice_365[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_510 = tensor.extract_slice %inserted_slice_503[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_511 = tensor.reshape %extracted_slice_510(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %439 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_511 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %440 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_508, %extracted_507, %extracted_slice_509 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%439 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_512 = tensor.collapse_shape %440 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_513 = tensor.insert_slice %collapsed_512 into %inserted_slice_503[672] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_514 = tensor.extract_slice %321#0[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_515 = tensor.extract_slice %extracted_slice_364[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %441 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_515, %extracted_slice_514 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %442 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%441 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %443 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %442) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %444 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%443 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_516 = tensor.extract %444[] : tensor<f32>
    %445 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%443, %extracted_516 : tensor<1024xf32>, f32) outs(%443 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %446 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%445 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_517 = tensor.extract %446[] : tensor<f32>
    %expanded_518 = tensor.expand_shape %445 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_519 = tensor.extract_slice %extracted_slice_365[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_520 = tensor.extract_slice %inserted_slice_513[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_521 = tensor.reshape %extracted_slice_520(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %447 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_521 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %448 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_518, %extracted_517, %extracted_slice_519 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%447 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_522 = tensor.collapse_shape %448 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_523 = tensor.insert_slice %collapsed_522 into %inserted_slice_513[720] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %449 = bufferization.materialize_in_destination %inserted_slice_523 in %315 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_524 = tensor.extract_slice %arg9[2, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %450 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_524, %449 : tensor<768x768xf32>, tensor<768xf32>) outs(%310#1 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<768xf32>
    %extracted_slice_525 = tensor.extract_slice %arg13[2, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %451 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%450 : tensor<768xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.mulf %in, %in : f32
      %917 = arith.addf %916, %out : f32
      linalg.yield %917 : f32
    } -> tensor<f32>
    %extracted_526 = tensor.extract %451[] : tensor<f32>
    %452 = arith.divf %extracted_526, %cst_3 : f32
    %453 = arith.addf %452, %cst_4 : f32
    %454 = math.rsqrt %453 : f32
    %455 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%450, %454, %extracted_slice_525 : tensor<768xf32>, f32, tensor<768xf32>) outs(%449 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      linalg.yield %917 : f32
    } -> tensor<768xf32>
    %456 = bufferization.materialize_in_destination %455 in %449 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_527 = tensor.extract_slice %arg10[2, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %extracted_slice_528 = tensor.extract_slice %arg12[2, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %457 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_527, %456 : tensor<2048x768xf32>, tensor<768xf32>) outs(%158 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<2048xf32>
    %458 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_528, %456 : tensor<2048x768xf32>, tensor<768xf32>) outs(%158 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<2048xf32>
    %extracted_slice_529 = tensor.extract_slice %arg11[2, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
    %459:2 = linalg.generic {indexing_maps = [#map3, #map4, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_529, %458 : tensor<768x2048xf32>, tensor<2048xf32>) outs(%457, %450 : tensor<2048xf32>, tensor<768xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32, %out_1059: f32):
      %916 = arith.negf %out : f32
      %917 = math.exp %916 : f32
      %918 = arith.addf %917, %cst_5 : f32
      %919 = arith.divf %cst_5, %918 : f32
      %920 = arith.mulf %out, %919 : f32
      %921 = arith.mulf %920, %in_1058 : f32
      %922 = arith.mulf %in, %921 : f32
      %923 = arith.addf %out_1059, %922 : f32
      linalg.yield %921, %923 : f32, f32
    } -> (tensor<2048xf32>, tensor<768xf32>)
    %extracted_slice_530 = tensor.extract_slice %arg5[3, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %460 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%459#1 : tensor<768xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.mulf %in, %in : f32
      %917 = arith.addf %916, %out : f32
      linalg.yield %917 : f32
    } -> tensor<f32>
    %extracted_531 = tensor.extract %460[] : tensor<f32>
    %461 = arith.divf %extracted_531, %cst_3 : f32
    %462 = arith.addf %461, %cst_4 : f32
    %463 = math.rsqrt %462 : f32
    %464 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%459#1, %463, %extracted_slice_530 : tensor<768xf32>, f32, tensor<768xf32>) outs(%6 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      linalg.yield %917 : f32
    } -> tensor<768xf32>
    %extracted_slice_532 = tensor.extract_slice %arg6[3, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_533 = tensor.extract_slice %arg7[3, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_534 = tensor.extract_slice %arg8[3, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %465 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_532, %464 : tensor<768x768xf32>, tensor<768xf32>) outs(%8 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<768xf32>
    %extracted_slice_535 = tensor.extract_slice %inserted_slice_363[3, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %466 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_535 : tensor<768xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<768xf32>
    %467 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_533, %464 : tensor<768x768xf32>, tensor<768xf32>) outs(%466 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<768xf32>
    %extracted_slice_536 = tensor.extract_slice %inserted_slice_362[3, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %468 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_536 : tensor<768xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<768xf32>
    %469 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_534, %464 : tensor<768x768xf32>, tensor<768xf32>) outs(%468 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<768xf32>
    %inserted_slice_537 = tensor.insert_slice %469 into %inserted_slice_362[3, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %470:2 = scf.for %arg16 = %c0 to %c768 step %c2 iter_args(%arg17 = %465, %arg18 = %467) -> (tensor<768xf32>, tensor<768xf32>) {
      %916 = arith.remui %arg16, %c48 : index
      %917 = arith.index_cast %916 : index to i64
      %918 = arith.uitofp %917 : i64 to f32
      %919 = arith.divf %918, %cst_6 : f32
      %920 = math.powf %cst_7, %919 : f32
      %921 = arith.divf %cst_5, %920 : f32
      %922 = arith.mulf %15, %921 : f32
      %923 = math.cos %922 : f32
      %924 = math.sin %922 : f32
      %925 = arith.addi %arg16, %c1 : index
      %extracted_1058 = tensor.extract %arg17[%arg16] : tensor<768xf32>
      %extracted_1059 = tensor.extract %arg17[%925] : tensor<768xf32>
      %926 = arith.mulf %extracted_1058, %923 : f32
      %927 = arith.mulf %extracted_1059, %924 : f32
      %928 = arith.subf %926, %927 : f32
      %inserted = tensor.insert %928 into %arg17[%arg16] : tensor<768xf32>
      %929 = arith.mulf %extracted_1058, %924 : f32
      %930 = arith.mulf %extracted_1059, %923 : f32
      %931 = arith.addf %929, %930 : f32
      %inserted_1060 = tensor.insert %931 into %inserted[%925] : tensor<768xf32>
      %932 = bufferization.materialize_in_destination %inserted_1060 in %arg17 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      %933 = arith.cmpi ult, %arg16, %c768 : index
      %934 = scf.if %933 -> (tensor<768xf32>) {
        %extracted_1061 = tensor.extract %arg18[%arg16] : tensor<768xf32>
        %extracted_1062 = tensor.extract %arg18[%925] : tensor<768xf32>
        %935 = arith.mulf %extracted_1061, %923 : f32
        %936 = arith.mulf %extracted_1062, %924 : f32
        %937 = arith.subf %935, %936 : f32
        %inserted_1063 = tensor.insert %937 into %arg18[%arg16] : tensor<768xf32>
        %938 = arith.mulf %extracted_1061, %924 : f32
        %939 = arith.mulf %extracted_1062, %923 : f32
        %940 = arith.addf %938, %939 : f32
        %inserted_1064 = tensor.insert %940 into %inserted_1063[%925] : tensor<768xf32>
        %941 = bufferization.materialize_in_destination %inserted_1064 in %arg18 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
        scf.yield %941 : tensor<768xf32>
      } else {
        scf.yield %arg18 : tensor<768xf32>
      }
      scf.yield %932, %934 : tensor<768xf32>, tensor<768xf32>
    }
    %inserted_slice_538 = tensor.insert_slice %470#1 into %inserted_slice_363[3, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %extracted_slice_539 = tensor.extract_slice %inserted_slice_538[3, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %extracted_slice_540 = tensor.extract_slice %inserted_slice_537[3, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %extracted_slice_541 = tensor.extract_slice %470#0[0] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_542 = tensor.extract_slice %extracted_slice_539[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %471 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_542, %extracted_slice_541 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %472 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%471 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %473 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %472) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %474 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%473 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_543 = tensor.extract %474[] : tensor<f32>
    %475 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%473, %extracted_543 : tensor<1024xf32>, f32) outs(%473 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %476 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%475 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_544 = tensor.extract %476[] : tensor<f32>
    %expanded_545 = tensor.expand_shape %475 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_546 = tensor.extract_slice %extracted_slice_540[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %477 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_545, %extracted_544, %extracted_slice_546 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%27 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_547 = tensor.collapse_shape %477 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_548 = tensor.insert_slice %collapsed_547 into %464[0] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_549 = tensor.extract_slice %470#0[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_550 = tensor.extract_slice %extracted_slice_539[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %478 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_550, %extracted_slice_549 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %479 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%478 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %480 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %479) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %481 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%480 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_551 = tensor.extract %481[] : tensor<f32>
    %482 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%480, %extracted_551 : tensor<1024xf32>, f32) outs(%480 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %483 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%482 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_552 = tensor.extract %483[] : tensor<f32>
    %expanded_553 = tensor.expand_shape %482 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_554 = tensor.extract_slice %extracted_slice_540[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_555 = tensor.extract_slice %inserted_slice_548[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_556 = tensor.reshape %extracted_slice_555(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %484 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_556 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %485 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_553, %extracted_552, %extracted_slice_554 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%484 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_557 = tensor.collapse_shape %485 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_558 = tensor.insert_slice %collapsed_557 into %inserted_slice_548[48] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_559 = tensor.extract_slice %470#0[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_560 = tensor.extract_slice %extracted_slice_539[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %486 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_560, %extracted_slice_559 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %487 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%486 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %488 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %487) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %489 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%488 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_561 = tensor.extract %489[] : tensor<f32>
    %490 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%488, %extracted_561 : tensor<1024xf32>, f32) outs(%488 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %491 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%490 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_562 = tensor.extract %491[] : tensor<f32>
    %expanded_563 = tensor.expand_shape %490 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_564 = tensor.extract_slice %extracted_slice_540[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_565 = tensor.extract_slice %inserted_slice_558[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_566 = tensor.reshape %extracted_slice_565(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %492 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_566 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %493 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_563, %extracted_562, %extracted_slice_564 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%492 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_567 = tensor.collapse_shape %493 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_568 = tensor.insert_slice %collapsed_567 into %inserted_slice_558[96] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_569 = tensor.extract_slice %470#0[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_570 = tensor.extract_slice %extracted_slice_539[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %494 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_570, %extracted_slice_569 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %495 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%494 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %496 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %495) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %497 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%496 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_571 = tensor.extract %497[] : tensor<f32>
    %498 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%496, %extracted_571 : tensor<1024xf32>, f32) outs(%496 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %499 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%498 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_572 = tensor.extract %499[] : tensor<f32>
    %expanded_573 = tensor.expand_shape %498 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_574 = tensor.extract_slice %extracted_slice_540[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_575 = tensor.extract_slice %inserted_slice_568[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_576 = tensor.reshape %extracted_slice_575(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %500 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_576 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %501 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_573, %extracted_572, %extracted_slice_574 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%500 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_577 = tensor.collapse_shape %501 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_578 = tensor.insert_slice %collapsed_577 into %inserted_slice_568[144] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_579 = tensor.extract_slice %470#0[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_580 = tensor.extract_slice %extracted_slice_539[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %502 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_580, %extracted_slice_579 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %503 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%502 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %504 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %503) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %505 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%504 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_581 = tensor.extract %505[] : tensor<f32>
    %506 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%504, %extracted_581 : tensor<1024xf32>, f32) outs(%504 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %507 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%506 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_582 = tensor.extract %507[] : tensor<f32>
    %expanded_583 = tensor.expand_shape %506 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_584 = tensor.extract_slice %extracted_slice_540[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_585 = tensor.extract_slice %inserted_slice_578[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_586 = tensor.reshape %extracted_slice_585(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %508 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_586 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %509 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_583, %extracted_582, %extracted_slice_584 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%508 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_587 = tensor.collapse_shape %509 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_588 = tensor.insert_slice %collapsed_587 into %inserted_slice_578[192] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_589 = tensor.extract_slice %470#0[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_590 = tensor.extract_slice %extracted_slice_539[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %510 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_590, %extracted_slice_589 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %511 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%510 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %512 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %511) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %513 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%512 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_591 = tensor.extract %513[] : tensor<f32>
    %514 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%512, %extracted_591 : tensor<1024xf32>, f32) outs(%512 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %515 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%514 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_592 = tensor.extract %515[] : tensor<f32>
    %expanded_593 = tensor.expand_shape %514 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_594 = tensor.extract_slice %extracted_slice_540[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_595 = tensor.extract_slice %inserted_slice_588[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_596 = tensor.reshape %extracted_slice_595(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %516 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_596 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %517 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_593, %extracted_592, %extracted_slice_594 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%516 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_597 = tensor.collapse_shape %517 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_598 = tensor.insert_slice %collapsed_597 into %inserted_slice_588[240] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_599 = tensor.extract_slice %470#0[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_600 = tensor.extract_slice %extracted_slice_539[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %518 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_600, %extracted_slice_599 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %519 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%518 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %520 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %519) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %521 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%520 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_601 = tensor.extract %521[] : tensor<f32>
    %522 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%520, %extracted_601 : tensor<1024xf32>, f32) outs(%520 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %523 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%522 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_602 = tensor.extract %523[] : tensor<f32>
    %expanded_603 = tensor.expand_shape %522 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_604 = tensor.extract_slice %extracted_slice_540[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_605 = tensor.extract_slice %inserted_slice_598[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_606 = tensor.reshape %extracted_slice_605(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %524 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_606 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %525 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_603, %extracted_602, %extracted_slice_604 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%524 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_607 = tensor.collapse_shape %525 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_608 = tensor.insert_slice %collapsed_607 into %inserted_slice_598[288] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_609 = tensor.extract_slice %470#0[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_610 = tensor.extract_slice %extracted_slice_539[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %526 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_610, %extracted_slice_609 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %527 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%526 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %528 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %527) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %529 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%528 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_611 = tensor.extract %529[] : tensor<f32>
    %530 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%528, %extracted_611 : tensor<1024xf32>, f32) outs(%528 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %531 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%530 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_612 = tensor.extract %531[] : tensor<f32>
    %expanded_613 = tensor.expand_shape %530 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_614 = tensor.extract_slice %extracted_slice_540[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_615 = tensor.extract_slice %inserted_slice_608[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_616 = tensor.reshape %extracted_slice_615(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %532 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_616 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %533 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_613, %extracted_612, %extracted_slice_614 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%532 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_617 = tensor.collapse_shape %533 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_618 = tensor.insert_slice %collapsed_617 into %inserted_slice_608[336] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_619 = tensor.extract_slice %470#0[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_620 = tensor.extract_slice %extracted_slice_539[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %534 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_620, %extracted_slice_619 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %535 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%534 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %536 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %535) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %537 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%536 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_621 = tensor.extract %537[] : tensor<f32>
    %538 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%536, %extracted_621 : tensor<1024xf32>, f32) outs(%536 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %539 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%538 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_622 = tensor.extract %539[] : tensor<f32>
    %expanded_623 = tensor.expand_shape %538 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_624 = tensor.extract_slice %extracted_slice_540[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_625 = tensor.extract_slice %inserted_slice_618[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_626 = tensor.reshape %extracted_slice_625(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %540 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_626 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %541 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_623, %extracted_622, %extracted_slice_624 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%540 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_627 = tensor.collapse_shape %541 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_628 = tensor.insert_slice %collapsed_627 into %inserted_slice_618[384] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_629 = tensor.extract_slice %470#0[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_630 = tensor.extract_slice %extracted_slice_539[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %542 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_630, %extracted_slice_629 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %543 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%542 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %544 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %543) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %545 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%544 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_631 = tensor.extract %545[] : tensor<f32>
    %546 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%544, %extracted_631 : tensor<1024xf32>, f32) outs(%544 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %547 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%546 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_632 = tensor.extract %547[] : tensor<f32>
    %expanded_633 = tensor.expand_shape %546 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_634 = tensor.extract_slice %extracted_slice_540[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_635 = tensor.extract_slice %inserted_slice_628[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_636 = tensor.reshape %extracted_slice_635(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %548 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_636 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %549 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_633, %extracted_632, %extracted_slice_634 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%548 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_637 = tensor.collapse_shape %549 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_638 = tensor.insert_slice %collapsed_637 into %inserted_slice_628[432] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_639 = tensor.extract_slice %470#0[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_640 = tensor.extract_slice %extracted_slice_539[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %550 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_640, %extracted_slice_639 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %551 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%550 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %552 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %551) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %553 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%552 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_641 = tensor.extract %553[] : tensor<f32>
    %554 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%552, %extracted_641 : tensor<1024xf32>, f32) outs(%552 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %555 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%554 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_642 = tensor.extract %555[] : tensor<f32>
    %expanded_643 = tensor.expand_shape %554 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_644 = tensor.extract_slice %extracted_slice_540[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_645 = tensor.extract_slice %inserted_slice_638[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_646 = tensor.reshape %extracted_slice_645(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %556 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_646 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %557 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_643, %extracted_642, %extracted_slice_644 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%556 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_647 = tensor.collapse_shape %557 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_648 = tensor.insert_slice %collapsed_647 into %inserted_slice_638[480] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_649 = tensor.extract_slice %470#0[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_650 = tensor.extract_slice %extracted_slice_539[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %558 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_650, %extracted_slice_649 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %559 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%558 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %560 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %559) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %561 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%560 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_651 = tensor.extract %561[] : tensor<f32>
    %562 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%560, %extracted_651 : tensor<1024xf32>, f32) outs(%560 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %563 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%562 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_652 = tensor.extract %563[] : tensor<f32>
    %expanded_653 = tensor.expand_shape %562 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_654 = tensor.extract_slice %extracted_slice_540[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_655 = tensor.extract_slice %inserted_slice_648[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_656 = tensor.reshape %extracted_slice_655(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %564 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_656 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %565 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_653, %extracted_652, %extracted_slice_654 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%564 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_657 = tensor.collapse_shape %565 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_658 = tensor.insert_slice %collapsed_657 into %inserted_slice_648[528] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_659 = tensor.extract_slice %470#0[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_660 = tensor.extract_slice %extracted_slice_539[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %566 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_660, %extracted_slice_659 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %567 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%566 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %568 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %567) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %569 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%568 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_661 = tensor.extract %569[] : tensor<f32>
    %570 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%568, %extracted_661 : tensor<1024xf32>, f32) outs(%568 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %571 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%570 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_662 = tensor.extract %571[] : tensor<f32>
    %expanded_663 = tensor.expand_shape %570 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_664 = tensor.extract_slice %extracted_slice_540[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_665 = tensor.extract_slice %inserted_slice_658[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_666 = tensor.reshape %extracted_slice_665(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %572 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_666 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %573 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_663, %extracted_662, %extracted_slice_664 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%572 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_667 = tensor.collapse_shape %573 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_668 = tensor.insert_slice %collapsed_667 into %inserted_slice_658[576] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_669 = tensor.extract_slice %470#0[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_670 = tensor.extract_slice %extracted_slice_539[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %574 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_670, %extracted_slice_669 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %575 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%574 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %576 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %575) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %577 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%576 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_671 = tensor.extract %577[] : tensor<f32>
    %578 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%576, %extracted_671 : tensor<1024xf32>, f32) outs(%576 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %579 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%578 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_672 = tensor.extract %579[] : tensor<f32>
    %expanded_673 = tensor.expand_shape %578 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_674 = tensor.extract_slice %extracted_slice_540[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_675 = tensor.extract_slice %inserted_slice_668[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_676 = tensor.reshape %extracted_slice_675(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %580 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_676 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %581 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_673, %extracted_672, %extracted_slice_674 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%580 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_677 = tensor.collapse_shape %581 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_678 = tensor.insert_slice %collapsed_677 into %inserted_slice_668[624] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_679 = tensor.extract_slice %470#0[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_680 = tensor.extract_slice %extracted_slice_539[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %582 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_680, %extracted_slice_679 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %583 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%582 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %584 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %583) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %585 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%584 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_681 = tensor.extract %585[] : tensor<f32>
    %586 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%584, %extracted_681 : tensor<1024xf32>, f32) outs(%584 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %587 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%586 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_682 = tensor.extract %587[] : tensor<f32>
    %expanded_683 = tensor.expand_shape %586 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_684 = tensor.extract_slice %extracted_slice_540[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_685 = tensor.extract_slice %inserted_slice_678[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_686 = tensor.reshape %extracted_slice_685(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %588 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_686 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %589 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_683, %extracted_682, %extracted_slice_684 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%588 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_687 = tensor.collapse_shape %589 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_688 = tensor.insert_slice %collapsed_687 into %inserted_slice_678[672] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_689 = tensor.extract_slice %470#0[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_690 = tensor.extract_slice %extracted_slice_539[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %590 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_690, %extracted_slice_689 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %591 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%590 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %592 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %591) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %593 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%592 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_691 = tensor.extract %593[] : tensor<f32>
    %594 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%592, %extracted_691 : tensor<1024xf32>, f32) outs(%592 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %595 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%594 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_692 = tensor.extract %595[] : tensor<f32>
    %expanded_693 = tensor.expand_shape %594 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_694 = tensor.extract_slice %extracted_slice_540[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_695 = tensor.extract_slice %inserted_slice_688[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_696 = tensor.reshape %extracted_slice_695(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %596 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_696 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %597 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_693, %extracted_692, %extracted_slice_694 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%596 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_697 = tensor.collapse_shape %597 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_698 = tensor.insert_slice %collapsed_697 into %inserted_slice_688[720] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %598 = bufferization.materialize_in_destination %inserted_slice_698 in %464 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_699 = tensor.extract_slice %arg9[3, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %599 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_699, %598 : tensor<768x768xf32>, tensor<768xf32>) outs(%459#1 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<768xf32>
    %extracted_slice_700 = tensor.extract_slice %arg13[3, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %600 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%599 : tensor<768xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.mulf %in, %in : f32
      %917 = arith.addf %916, %out : f32
      linalg.yield %917 : f32
    } -> tensor<f32>
    %extracted_701 = tensor.extract %600[] : tensor<f32>
    %601 = arith.divf %extracted_701, %cst_3 : f32
    %602 = arith.addf %601, %cst_4 : f32
    %603 = math.rsqrt %602 : f32
    %604 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%599, %603, %extracted_slice_700 : tensor<768xf32>, f32, tensor<768xf32>) outs(%598 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      linalg.yield %917 : f32
    } -> tensor<768xf32>
    %605 = bufferization.materialize_in_destination %604 in %598 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_702 = tensor.extract_slice %arg10[3, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %extracted_slice_703 = tensor.extract_slice %arg12[3, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %606 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_702, %605 : tensor<2048x768xf32>, tensor<768xf32>) outs(%158 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<2048xf32>
    %607 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_703, %605 : tensor<2048x768xf32>, tensor<768xf32>) outs(%158 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<2048xf32>
    %extracted_slice_704 = tensor.extract_slice %arg11[3, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
    %608:2 = linalg.generic {indexing_maps = [#map3, #map4, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_704, %607 : tensor<768x2048xf32>, tensor<2048xf32>) outs(%606, %599 : tensor<2048xf32>, tensor<768xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32, %out_1059: f32):
      %916 = arith.negf %out : f32
      %917 = math.exp %916 : f32
      %918 = arith.addf %917, %cst_5 : f32
      %919 = arith.divf %cst_5, %918 : f32
      %920 = arith.mulf %out, %919 : f32
      %921 = arith.mulf %920, %in_1058 : f32
      %922 = arith.mulf %in, %921 : f32
      %923 = arith.addf %out_1059, %922 : f32
      linalg.yield %921, %923 : f32, f32
    } -> (tensor<2048xf32>, tensor<768xf32>)
    %extracted_slice_705 = tensor.extract_slice %arg5[4, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %609 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%608#1 : tensor<768xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.mulf %in, %in : f32
      %917 = arith.addf %916, %out : f32
      linalg.yield %917 : f32
    } -> tensor<f32>
    %extracted_706 = tensor.extract %609[] : tensor<f32>
    %610 = arith.divf %extracted_706, %cst_3 : f32
    %611 = arith.addf %610, %cst_4 : f32
    %612 = math.rsqrt %611 : f32
    %613 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%608#1, %612, %extracted_slice_705 : tensor<768xf32>, f32, tensor<768xf32>) outs(%6 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      linalg.yield %917 : f32
    } -> tensor<768xf32>
    %extracted_slice_707 = tensor.extract_slice %arg6[4, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_708 = tensor.extract_slice %arg7[4, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_709 = tensor.extract_slice %arg8[4, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %614 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_707, %613 : tensor<768x768xf32>, tensor<768xf32>) outs(%8 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<768xf32>
    %extracted_slice_710 = tensor.extract_slice %inserted_slice_538[4, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %615 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_710 : tensor<768xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<768xf32>
    %616 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_708, %613 : tensor<768x768xf32>, tensor<768xf32>) outs(%615 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<768xf32>
    %extracted_slice_711 = tensor.extract_slice %inserted_slice_537[4, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %617 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_711 : tensor<768xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<768xf32>
    %618 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_709, %613 : tensor<768x768xf32>, tensor<768xf32>) outs(%617 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<768xf32>
    %inserted_slice_712 = tensor.insert_slice %618 into %inserted_slice_537[4, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %619:2 = scf.for %arg16 = %c0 to %c768 step %c2 iter_args(%arg17 = %614, %arg18 = %616) -> (tensor<768xf32>, tensor<768xf32>) {
      %916 = arith.remui %arg16, %c48 : index
      %917 = arith.index_cast %916 : index to i64
      %918 = arith.uitofp %917 : i64 to f32
      %919 = arith.divf %918, %cst_6 : f32
      %920 = math.powf %cst_7, %919 : f32
      %921 = arith.divf %cst_5, %920 : f32
      %922 = arith.mulf %15, %921 : f32
      %923 = math.cos %922 : f32
      %924 = math.sin %922 : f32
      %925 = arith.addi %arg16, %c1 : index
      %extracted_1058 = tensor.extract %arg17[%arg16] : tensor<768xf32>
      %extracted_1059 = tensor.extract %arg17[%925] : tensor<768xf32>
      %926 = arith.mulf %extracted_1058, %923 : f32
      %927 = arith.mulf %extracted_1059, %924 : f32
      %928 = arith.subf %926, %927 : f32
      %inserted = tensor.insert %928 into %arg17[%arg16] : tensor<768xf32>
      %929 = arith.mulf %extracted_1058, %924 : f32
      %930 = arith.mulf %extracted_1059, %923 : f32
      %931 = arith.addf %929, %930 : f32
      %inserted_1060 = tensor.insert %931 into %inserted[%925] : tensor<768xf32>
      %932 = bufferization.materialize_in_destination %inserted_1060 in %arg17 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      %933 = arith.cmpi ult, %arg16, %c768 : index
      %934 = scf.if %933 -> (tensor<768xf32>) {
        %extracted_1061 = tensor.extract %arg18[%arg16] : tensor<768xf32>
        %extracted_1062 = tensor.extract %arg18[%925] : tensor<768xf32>
        %935 = arith.mulf %extracted_1061, %923 : f32
        %936 = arith.mulf %extracted_1062, %924 : f32
        %937 = arith.subf %935, %936 : f32
        %inserted_1063 = tensor.insert %937 into %arg18[%arg16] : tensor<768xf32>
        %938 = arith.mulf %extracted_1061, %924 : f32
        %939 = arith.mulf %extracted_1062, %923 : f32
        %940 = arith.addf %938, %939 : f32
        %inserted_1064 = tensor.insert %940 into %inserted_1063[%925] : tensor<768xf32>
        %941 = bufferization.materialize_in_destination %inserted_1064 in %arg18 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
        scf.yield %941 : tensor<768xf32>
      } else {
        scf.yield %arg18 : tensor<768xf32>
      }
      scf.yield %932, %934 : tensor<768xf32>, tensor<768xf32>
    }
    %inserted_slice_713 = tensor.insert_slice %619#1 into %inserted_slice_538[4, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %extracted_slice_714 = tensor.extract_slice %inserted_slice_713[4, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %extracted_slice_715 = tensor.extract_slice %inserted_slice_712[4, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %extracted_slice_716 = tensor.extract_slice %619#0[0] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_717 = tensor.extract_slice %extracted_slice_714[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %620 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_717, %extracted_slice_716 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %621 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%620 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %622 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %621) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %623 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%622 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_718 = tensor.extract %623[] : tensor<f32>
    %624 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%622, %extracted_718 : tensor<1024xf32>, f32) outs(%622 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %625 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%624 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_719 = tensor.extract %625[] : tensor<f32>
    %expanded_720 = tensor.expand_shape %624 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_721 = tensor.extract_slice %extracted_slice_715[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %626 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_720, %extracted_719, %extracted_slice_721 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%27 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_722 = tensor.collapse_shape %626 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_723 = tensor.insert_slice %collapsed_722 into %613[0] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_724 = tensor.extract_slice %619#0[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_725 = tensor.extract_slice %extracted_slice_714[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %627 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_725, %extracted_slice_724 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %628 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%627 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %629 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %628) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %630 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%629 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_726 = tensor.extract %630[] : tensor<f32>
    %631 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%629, %extracted_726 : tensor<1024xf32>, f32) outs(%629 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %632 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%631 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_727 = tensor.extract %632[] : tensor<f32>
    %expanded_728 = tensor.expand_shape %631 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_729 = tensor.extract_slice %extracted_slice_715[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_730 = tensor.extract_slice %inserted_slice_723[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_731 = tensor.reshape %extracted_slice_730(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %633 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_731 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %634 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_728, %extracted_727, %extracted_slice_729 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%633 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_732 = tensor.collapse_shape %634 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_733 = tensor.insert_slice %collapsed_732 into %inserted_slice_723[48] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_734 = tensor.extract_slice %619#0[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_735 = tensor.extract_slice %extracted_slice_714[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %635 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_735, %extracted_slice_734 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %636 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%635 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %637 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %636) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %638 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%637 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_736 = tensor.extract %638[] : tensor<f32>
    %639 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%637, %extracted_736 : tensor<1024xf32>, f32) outs(%637 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %640 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%639 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_737 = tensor.extract %640[] : tensor<f32>
    %expanded_738 = tensor.expand_shape %639 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_739 = tensor.extract_slice %extracted_slice_715[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_740 = tensor.extract_slice %inserted_slice_733[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_741 = tensor.reshape %extracted_slice_740(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %641 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_741 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %642 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_738, %extracted_737, %extracted_slice_739 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%641 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_742 = tensor.collapse_shape %642 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_743 = tensor.insert_slice %collapsed_742 into %inserted_slice_733[96] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_744 = tensor.extract_slice %619#0[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_745 = tensor.extract_slice %extracted_slice_714[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %643 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_745, %extracted_slice_744 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %644 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%643 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %645 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %644) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %646 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%645 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_746 = tensor.extract %646[] : tensor<f32>
    %647 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%645, %extracted_746 : tensor<1024xf32>, f32) outs(%645 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %648 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%647 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_747 = tensor.extract %648[] : tensor<f32>
    %expanded_748 = tensor.expand_shape %647 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_749 = tensor.extract_slice %extracted_slice_715[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_750 = tensor.extract_slice %inserted_slice_743[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_751 = tensor.reshape %extracted_slice_750(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %649 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_751 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %650 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_748, %extracted_747, %extracted_slice_749 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%649 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_752 = tensor.collapse_shape %650 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_753 = tensor.insert_slice %collapsed_752 into %inserted_slice_743[144] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_754 = tensor.extract_slice %619#0[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_755 = tensor.extract_slice %extracted_slice_714[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %651 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_755, %extracted_slice_754 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %652 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%651 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %653 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %652) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %654 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%653 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_756 = tensor.extract %654[] : tensor<f32>
    %655 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%653, %extracted_756 : tensor<1024xf32>, f32) outs(%653 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %656 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%655 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_757 = tensor.extract %656[] : tensor<f32>
    %expanded_758 = tensor.expand_shape %655 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_759 = tensor.extract_slice %extracted_slice_715[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_760 = tensor.extract_slice %inserted_slice_753[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_761 = tensor.reshape %extracted_slice_760(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %657 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_761 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %658 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_758, %extracted_757, %extracted_slice_759 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%657 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_762 = tensor.collapse_shape %658 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_763 = tensor.insert_slice %collapsed_762 into %inserted_slice_753[192] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_764 = tensor.extract_slice %619#0[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_765 = tensor.extract_slice %extracted_slice_714[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %659 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_765, %extracted_slice_764 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %660 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%659 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %661 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %660) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %662 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%661 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_766 = tensor.extract %662[] : tensor<f32>
    %663 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%661, %extracted_766 : tensor<1024xf32>, f32) outs(%661 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %664 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%663 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_767 = tensor.extract %664[] : tensor<f32>
    %expanded_768 = tensor.expand_shape %663 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_769 = tensor.extract_slice %extracted_slice_715[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_770 = tensor.extract_slice %inserted_slice_763[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_771 = tensor.reshape %extracted_slice_770(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %665 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_771 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %666 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_768, %extracted_767, %extracted_slice_769 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%665 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_772 = tensor.collapse_shape %666 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_773 = tensor.insert_slice %collapsed_772 into %inserted_slice_763[240] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_774 = tensor.extract_slice %619#0[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_775 = tensor.extract_slice %extracted_slice_714[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %667 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_775, %extracted_slice_774 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %668 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%667 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %669 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %668) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %670 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%669 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_776 = tensor.extract %670[] : tensor<f32>
    %671 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%669, %extracted_776 : tensor<1024xf32>, f32) outs(%669 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %672 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%671 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_777 = tensor.extract %672[] : tensor<f32>
    %expanded_778 = tensor.expand_shape %671 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_779 = tensor.extract_slice %extracted_slice_715[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_780 = tensor.extract_slice %inserted_slice_773[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_781 = tensor.reshape %extracted_slice_780(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %673 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_781 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %674 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_778, %extracted_777, %extracted_slice_779 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%673 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_782 = tensor.collapse_shape %674 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_783 = tensor.insert_slice %collapsed_782 into %inserted_slice_773[288] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_784 = tensor.extract_slice %619#0[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_785 = tensor.extract_slice %extracted_slice_714[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %675 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_785, %extracted_slice_784 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %676 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%675 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %677 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %676) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %678 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%677 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_786 = tensor.extract %678[] : tensor<f32>
    %679 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%677, %extracted_786 : tensor<1024xf32>, f32) outs(%677 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %680 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%679 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_787 = tensor.extract %680[] : tensor<f32>
    %expanded_788 = tensor.expand_shape %679 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_789 = tensor.extract_slice %extracted_slice_715[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_790 = tensor.extract_slice %inserted_slice_783[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_791 = tensor.reshape %extracted_slice_790(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %681 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_791 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %682 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_788, %extracted_787, %extracted_slice_789 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%681 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_792 = tensor.collapse_shape %682 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_793 = tensor.insert_slice %collapsed_792 into %inserted_slice_783[336] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_794 = tensor.extract_slice %619#0[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_795 = tensor.extract_slice %extracted_slice_714[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %683 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_795, %extracted_slice_794 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %684 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%683 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %685 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %684) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %686 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%685 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_796 = tensor.extract %686[] : tensor<f32>
    %687 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%685, %extracted_796 : tensor<1024xf32>, f32) outs(%685 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %688 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%687 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_797 = tensor.extract %688[] : tensor<f32>
    %expanded_798 = tensor.expand_shape %687 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_799 = tensor.extract_slice %extracted_slice_715[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_800 = tensor.extract_slice %inserted_slice_793[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_801 = tensor.reshape %extracted_slice_800(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %689 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_801 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %690 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_798, %extracted_797, %extracted_slice_799 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%689 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_802 = tensor.collapse_shape %690 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_803 = tensor.insert_slice %collapsed_802 into %inserted_slice_793[384] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_804 = tensor.extract_slice %619#0[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_805 = tensor.extract_slice %extracted_slice_714[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %691 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_805, %extracted_slice_804 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %692 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%691 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %693 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %692) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %694 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%693 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_806 = tensor.extract %694[] : tensor<f32>
    %695 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%693, %extracted_806 : tensor<1024xf32>, f32) outs(%693 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %696 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%695 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_807 = tensor.extract %696[] : tensor<f32>
    %expanded_808 = tensor.expand_shape %695 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_809 = tensor.extract_slice %extracted_slice_715[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_810 = tensor.extract_slice %inserted_slice_803[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_811 = tensor.reshape %extracted_slice_810(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %697 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_811 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %698 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_808, %extracted_807, %extracted_slice_809 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%697 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_812 = tensor.collapse_shape %698 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_813 = tensor.insert_slice %collapsed_812 into %inserted_slice_803[432] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_814 = tensor.extract_slice %619#0[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_815 = tensor.extract_slice %extracted_slice_714[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %699 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_815, %extracted_slice_814 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %700 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%699 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %701 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %700) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %702 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%701 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_816 = tensor.extract %702[] : tensor<f32>
    %703 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%701, %extracted_816 : tensor<1024xf32>, f32) outs(%701 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %704 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%703 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_817 = tensor.extract %704[] : tensor<f32>
    %expanded_818 = tensor.expand_shape %703 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_819 = tensor.extract_slice %extracted_slice_715[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_820 = tensor.extract_slice %inserted_slice_813[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_821 = tensor.reshape %extracted_slice_820(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %705 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_821 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %706 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_818, %extracted_817, %extracted_slice_819 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%705 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_822 = tensor.collapse_shape %706 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_823 = tensor.insert_slice %collapsed_822 into %inserted_slice_813[480] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_824 = tensor.extract_slice %619#0[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_825 = tensor.extract_slice %extracted_slice_714[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %707 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_825, %extracted_slice_824 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %708 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%707 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %709 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %708) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %710 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%709 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_826 = tensor.extract %710[] : tensor<f32>
    %711 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%709, %extracted_826 : tensor<1024xf32>, f32) outs(%709 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %712 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%711 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_827 = tensor.extract %712[] : tensor<f32>
    %expanded_828 = tensor.expand_shape %711 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_829 = tensor.extract_slice %extracted_slice_715[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_830 = tensor.extract_slice %inserted_slice_823[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_831 = tensor.reshape %extracted_slice_830(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %713 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_831 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %714 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_828, %extracted_827, %extracted_slice_829 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%713 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_832 = tensor.collapse_shape %714 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_833 = tensor.insert_slice %collapsed_832 into %inserted_slice_823[528] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_834 = tensor.extract_slice %619#0[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_835 = tensor.extract_slice %extracted_slice_714[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %715 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_835, %extracted_slice_834 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %716 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%715 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %717 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %716) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %718 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%717 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_836 = tensor.extract %718[] : tensor<f32>
    %719 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%717, %extracted_836 : tensor<1024xf32>, f32) outs(%717 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %720 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%719 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_837 = tensor.extract %720[] : tensor<f32>
    %expanded_838 = tensor.expand_shape %719 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_839 = tensor.extract_slice %extracted_slice_715[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_840 = tensor.extract_slice %inserted_slice_833[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_841 = tensor.reshape %extracted_slice_840(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %721 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_841 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %722 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_838, %extracted_837, %extracted_slice_839 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%721 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_842 = tensor.collapse_shape %722 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_843 = tensor.insert_slice %collapsed_842 into %inserted_slice_833[576] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_844 = tensor.extract_slice %619#0[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_845 = tensor.extract_slice %extracted_slice_714[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %723 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_845, %extracted_slice_844 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %724 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%723 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %725 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %724) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %726 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%725 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_846 = tensor.extract %726[] : tensor<f32>
    %727 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%725, %extracted_846 : tensor<1024xf32>, f32) outs(%725 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %728 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%727 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_847 = tensor.extract %728[] : tensor<f32>
    %expanded_848 = tensor.expand_shape %727 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_849 = tensor.extract_slice %extracted_slice_715[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_850 = tensor.extract_slice %inserted_slice_843[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_851 = tensor.reshape %extracted_slice_850(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %729 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_851 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %730 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_848, %extracted_847, %extracted_slice_849 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%729 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_852 = tensor.collapse_shape %730 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_853 = tensor.insert_slice %collapsed_852 into %inserted_slice_843[624] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_854 = tensor.extract_slice %619#0[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_855 = tensor.extract_slice %extracted_slice_714[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %731 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_855, %extracted_slice_854 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %732 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%731 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %733 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %732) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %734 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%733 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_856 = tensor.extract %734[] : tensor<f32>
    %735 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%733, %extracted_856 : tensor<1024xf32>, f32) outs(%733 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %736 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%735 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_857 = tensor.extract %736[] : tensor<f32>
    %expanded_858 = tensor.expand_shape %735 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_859 = tensor.extract_slice %extracted_slice_715[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_860 = tensor.extract_slice %inserted_slice_853[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_861 = tensor.reshape %extracted_slice_860(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %737 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_861 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %738 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_858, %extracted_857, %extracted_slice_859 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%737 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_862 = tensor.collapse_shape %738 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_863 = tensor.insert_slice %collapsed_862 into %inserted_slice_853[672] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_864 = tensor.extract_slice %619#0[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_865 = tensor.extract_slice %extracted_slice_714[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %739 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_865, %extracted_slice_864 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %740 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%739 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %741 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %740) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %742 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%741 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_866 = tensor.extract %742[] : tensor<f32>
    %743 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%741, %extracted_866 : tensor<1024xf32>, f32) outs(%741 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %744 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%743 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_867 = tensor.extract %744[] : tensor<f32>
    %expanded_868 = tensor.expand_shape %743 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_869 = tensor.extract_slice %extracted_slice_715[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_870 = tensor.extract_slice %inserted_slice_863[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_871 = tensor.reshape %extracted_slice_870(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %745 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_871 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %746 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_868, %extracted_867, %extracted_slice_869 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%745 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_872 = tensor.collapse_shape %746 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_873 = tensor.insert_slice %collapsed_872 into %inserted_slice_863[720] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %747 = bufferization.materialize_in_destination %inserted_slice_873 in %613 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_874 = tensor.extract_slice %arg9[4, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %748 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_874, %747 : tensor<768x768xf32>, tensor<768xf32>) outs(%608#1 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<768xf32>
    %extracted_slice_875 = tensor.extract_slice %arg13[4, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %749 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%748 : tensor<768xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.mulf %in, %in : f32
      %917 = arith.addf %916, %out : f32
      linalg.yield %917 : f32
    } -> tensor<f32>
    %extracted_876 = tensor.extract %749[] : tensor<f32>
    %750 = arith.divf %extracted_876, %cst_3 : f32
    %751 = arith.addf %750, %cst_4 : f32
    %752 = math.rsqrt %751 : f32
    %753 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%748, %752, %extracted_slice_875 : tensor<768xf32>, f32, tensor<768xf32>) outs(%747 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      linalg.yield %917 : f32
    } -> tensor<768xf32>
    %754 = bufferization.materialize_in_destination %753 in %747 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_877 = tensor.extract_slice %arg10[4, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %extracted_slice_878 = tensor.extract_slice %arg12[4, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %755 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_877, %754 : tensor<2048x768xf32>, tensor<768xf32>) outs(%158 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<2048xf32>
    %756 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_878, %754 : tensor<2048x768xf32>, tensor<768xf32>) outs(%158 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<2048xf32>
    %extracted_slice_879 = tensor.extract_slice %arg11[4, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
    %757:2 = linalg.generic {indexing_maps = [#map3, #map4, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_879, %756 : tensor<768x2048xf32>, tensor<2048xf32>) outs(%755, %748 : tensor<2048xf32>, tensor<768xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32, %out_1059: f32):
      %916 = arith.negf %out : f32
      %917 = math.exp %916 : f32
      %918 = arith.addf %917, %cst_5 : f32
      %919 = arith.divf %cst_5, %918 : f32
      %920 = arith.mulf %out, %919 : f32
      %921 = arith.mulf %920, %in_1058 : f32
      %922 = arith.mulf %in, %921 : f32
      %923 = arith.addf %out_1059, %922 : f32
      linalg.yield %921, %923 : f32, f32
    } -> (tensor<2048xf32>, tensor<768xf32>)
    %extracted_slice_880 = tensor.extract_slice %arg5[5, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %758 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%757#1 : tensor<768xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.mulf %in, %in : f32
      %917 = arith.addf %916, %out : f32
      linalg.yield %917 : f32
    } -> tensor<f32>
    %extracted_881 = tensor.extract %758[] : tensor<f32>
    %759 = arith.divf %extracted_881, %cst_3 : f32
    %760 = arith.addf %759, %cst_4 : f32
    %761 = math.rsqrt %760 : f32
    %762 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%757#1, %761, %extracted_slice_880 : tensor<768xf32>, f32, tensor<768xf32>) outs(%6 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      linalg.yield %917 : f32
    } -> tensor<768xf32>
    %extracted_slice_882 = tensor.extract_slice %arg6[5, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_883 = tensor.extract_slice %arg7[5, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_884 = tensor.extract_slice %arg8[5, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %763 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_882, %762 : tensor<768x768xf32>, tensor<768xf32>) outs(%8 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<768xf32>
    %extracted_slice_885 = tensor.extract_slice %inserted_slice_713[5, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %764 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_885 : tensor<768xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<768xf32>
    %765 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_883, %762 : tensor<768x768xf32>, tensor<768xf32>) outs(%764 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<768xf32>
    %extracted_slice_886 = tensor.extract_slice %inserted_slice_712[5, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %766 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_886 : tensor<768xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<768xf32>
    %767 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_884, %762 : tensor<768x768xf32>, tensor<768xf32>) outs(%766 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<768xf32>
    %inserted_slice_887 = tensor.insert_slice %767 into %inserted_slice_712[5, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %768:2 = scf.for %arg16 = %c0 to %c768 step %c2 iter_args(%arg17 = %763, %arg18 = %765) -> (tensor<768xf32>, tensor<768xf32>) {
      %916 = arith.remui %arg16, %c48 : index
      %917 = arith.index_cast %916 : index to i64
      %918 = arith.uitofp %917 : i64 to f32
      %919 = arith.divf %918, %cst_6 : f32
      %920 = math.powf %cst_7, %919 : f32
      %921 = arith.divf %cst_5, %920 : f32
      %922 = arith.mulf %15, %921 : f32
      %923 = math.cos %922 : f32
      %924 = math.sin %922 : f32
      %925 = arith.addi %arg16, %c1 : index
      %extracted_1058 = tensor.extract %arg17[%arg16] : tensor<768xf32>
      %extracted_1059 = tensor.extract %arg17[%925] : tensor<768xf32>
      %926 = arith.mulf %extracted_1058, %923 : f32
      %927 = arith.mulf %extracted_1059, %924 : f32
      %928 = arith.subf %926, %927 : f32
      %inserted = tensor.insert %928 into %arg17[%arg16] : tensor<768xf32>
      %929 = arith.mulf %extracted_1058, %924 : f32
      %930 = arith.mulf %extracted_1059, %923 : f32
      %931 = arith.addf %929, %930 : f32
      %inserted_1060 = tensor.insert %931 into %inserted[%925] : tensor<768xf32>
      %932 = bufferization.materialize_in_destination %inserted_1060 in %arg17 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      %933 = arith.cmpi ult, %arg16, %c768 : index
      %934 = scf.if %933 -> (tensor<768xf32>) {
        %extracted_1061 = tensor.extract %arg18[%arg16] : tensor<768xf32>
        %extracted_1062 = tensor.extract %arg18[%925] : tensor<768xf32>
        %935 = arith.mulf %extracted_1061, %923 : f32
        %936 = arith.mulf %extracted_1062, %924 : f32
        %937 = arith.subf %935, %936 : f32
        %inserted_1063 = tensor.insert %937 into %arg18[%arg16] : tensor<768xf32>
        %938 = arith.mulf %extracted_1061, %924 : f32
        %939 = arith.mulf %extracted_1062, %923 : f32
        %940 = arith.addf %938, %939 : f32
        %inserted_1064 = tensor.insert %940 into %inserted_1063[%925] : tensor<768xf32>
        %941 = bufferization.materialize_in_destination %inserted_1064 in %arg18 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
        scf.yield %941 : tensor<768xf32>
      } else {
        scf.yield %arg18 : tensor<768xf32>
      }
      scf.yield %932, %934 : tensor<768xf32>, tensor<768xf32>
    }
    %inserted_slice_888 = tensor.insert_slice %768#1 into %inserted_slice_713[5, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %extracted_slice_889 = tensor.extract_slice %inserted_slice_888[5, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %extracted_slice_890 = tensor.extract_slice %inserted_slice_887[5, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %extracted_slice_891 = tensor.extract_slice %768#0[0] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_892 = tensor.extract_slice %extracted_slice_889[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %769 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_892, %extracted_slice_891 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %770 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%769 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %771 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %770) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %772 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%771 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_893 = tensor.extract %772[] : tensor<f32>
    %773 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%771, %extracted_893 : tensor<1024xf32>, f32) outs(%771 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %774 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%773 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_894 = tensor.extract %774[] : tensor<f32>
    %expanded_895 = tensor.expand_shape %773 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_896 = tensor.extract_slice %extracted_slice_890[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %775 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_895, %extracted_894, %extracted_slice_896 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%27 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_897 = tensor.collapse_shape %775 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_898 = tensor.insert_slice %collapsed_897 into %762[0] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_899 = tensor.extract_slice %768#0[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_900 = tensor.extract_slice %extracted_slice_889[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %776 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_900, %extracted_slice_899 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %777 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%776 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %778 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %777) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%778 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_901 = tensor.extract %779[] : tensor<f32>
    %780 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%778, %extracted_901 : tensor<1024xf32>, f32) outs(%778 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %781 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%780 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_902 = tensor.extract %781[] : tensor<f32>
    %expanded_903 = tensor.expand_shape %780 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_904 = tensor.extract_slice %extracted_slice_890[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_905 = tensor.extract_slice %inserted_slice_898[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_906 = tensor.reshape %extracted_slice_905(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %782 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_906 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %783 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_903, %extracted_902, %extracted_slice_904 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%782 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_907 = tensor.collapse_shape %783 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_908 = tensor.insert_slice %collapsed_907 into %inserted_slice_898[48] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_909 = tensor.extract_slice %768#0[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_910 = tensor.extract_slice %extracted_slice_889[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %784 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_910, %extracted_slice_909 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %785 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%784 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %786 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %785) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %787 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%786 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_911 = tensor.extract %787[] : tensor<f32>
    %788 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%786, %extracted_911 : tensor<1024xf32>, f32) outs(%786 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %789 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%788 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_912 = tensor.extract %789[] : tensor<f32>
    %expanded_913 = tensor.expand_shape %788 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_914 = tensor.extract_slice %extracted_slice_890[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_915 = tensor.extract_slice %inserted_slice_908[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_916 = tensor.reshape %extracted_slice_915(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %790 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_916 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %791 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_913, %extracted_912, %extracted_slice_914 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%790 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_917 = tensor.collapse_shape %791 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_918 = tensor.insert_slice %collapsed_917 into %inserted_slice_908[96] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_919 = tensor.extract_slice %768#0[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_920 = tensor.extract_slice %extracted_slice_889[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %792 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_920, %extracted_slice_919 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %793 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%792 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %794 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %793) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %795 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%794 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_921 = tensor.extract %795[] : tensor<f32>
    %796 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%794, %extracted_921 : tensor<1024xf32>, f32) outs(%794 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %797 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%796 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_922 = tensor.extract %797[] : tensor<f32>
    %expanded_923 = tensor.expand_shape %796 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_924 = tensor.extract_slice %extracted_slice_890[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_925 = tensor.extract_slice %inserted_slice_918[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_926 = tensor.reshape %extracted_slice_925(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %798 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_926 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %799 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_923, %extracted_922, %extracted_slice_924 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%798 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_927 = tensor.collapse_shape %799 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_928 = tensor.insert_slice %collapsed_927 into %inserted_slice_918[144] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_929 = tensor.extract_slice %768#0[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_930 = tensor.extract_slice %extracted_slice_889[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %800 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_930, %extracted_slice_929 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %801 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%800 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %802 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %801) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %803 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%802 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_931 = tensor.extract %803[] : tensor<f32>
    %804 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%802, %extracted_931 : tensor<1024xf32>, f32) outs(%802 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %805 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%804 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_932 = tensor.extract %805[] : tensor<f32>
    %expanded_933 = tensor.expand_shape %804 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_934 = tensor.extract_slice %extracted_slice_890[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_935 = tensor.extract_slice %inserted_slice_928[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_936 = tensor.reshape %extracted_slice_935(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %806 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_936 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %807 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_933, %extracted_932, %extracted_slice_934 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%806 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_937 = tensor.collapse_shape %807 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_938 = tensor.insert_slice %collapsed_937 into %inserted_slice_928[192] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_939 = tensor.extract_slice %768#0[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_940 = tensor.extract_slice %extracted_slice_889[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %808 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_940, %extracted_slice_939 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %809 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%808 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %810 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %809) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %811 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%810 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_941 = tensor.extract %811[] : tensor<f32>
    %812 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%810, %extracted_941 : tensor<1024xf32>, f32) outs(%810 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %813 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%812 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_942 = tensor.extract %813[] : tensor<f32>
    %expanded_943 = tensor.expand_shape %812 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_944 = tensor.extract_slice %extracted_slice_890[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_945 = tensor.extract_slice %inserted_slice_938[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_946 = tensor.reshape %extracted_slice_945(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %814 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_946 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %815 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_943, %extracted_942, %extracted_slice_944 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%814 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_947 = tensor.collapse_shape %815 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_948 = tensor.insert_slice %collapsed_947 into %inserted_slice_938[240] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_949 = tensor.extract_slice %768#0[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_950 = tensor.extract_slice %extracted_slice_889[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %816 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_950, %extracted_slice_949 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %817 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%816 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %818 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %817) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %819 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%818 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_951 = tensor.extract %819[] : tensor<f32>
    %820 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%818, %extracted_951 : tensor<1024xf32>, f32) outs(%818 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %821 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%820 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_952 = tensor.extract %821[] : tensor<f32>
    %expanded_953 = tensor.expand_shape %820 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_954 = tensor.extract_slice %extracted_slice_890[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_955 = tensor.extract_slice %inserted_slice_948[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_956 = tensor.reshape %extracted_slice_955(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %822 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_956 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %823 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_953, %extracted_952, %extracted_slice_954 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%822 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_957 = tensor.collapse_shape %823 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_958 = tensor.insert_slice %collapsed_957 into %inserted_slice_948[288] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_959 = tensor.extract_slice %768#0[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_960 = tensor.extract_slice %extracted_slice_889[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %824 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_960, %extracted_slice_959 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %825 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%824 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %826 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %825) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %827 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%826 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_961 = tensor.extract %827[] : tensor<f32>
    %828 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%826, %extracted_961 : tensor<1024xf32>, f32) outs(%826 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %829 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%828 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_962 = tensor.extract %829[] : tensor<f32>
    %expanded_963 = tensor.expand_shape %828 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_964 = tensor.extract_slice %extracted_slice_890[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_965 = tensor.extract_slice %inserted_slice_958[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_966 = tensor.reshape %extracted_slice_965(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %830 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_966 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %831 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_963, %extracted_962, %extracted_slice_964 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%830 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_967 = tensor.collapse_shape %831 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_968 = tensor.insert_slice %collapsed_967 into %inserted_slice_958[336] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_969 = tensor.extract_slice %768#0[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_970 = tensor.extract_slice %extracted_slice_889[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %832 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_970, %extracted_slice_969 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %833 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%832 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %834 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %833) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %835 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%834 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_971 = tensor.extract %835[] : tensor<f32>
    %836 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%834, %extracted_971 : tensor<1024xf32>, f32) outs(%834 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %837 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%836 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_972 = tensor.extract %837[] : tensor<f32>
    %expanded_973 = tensor.expand_shape %836 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_974 = tensor.extract_slice %extracted_slice_890[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_975 = tensor.extract_slice %inserted_slice_968[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_976 = tensor.reshape %extracted_slice_975(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %838 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_976 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %839 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_973, %extracted_972, %extracted_slice_974 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%838 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_977 = tensor.collapse_shape %839 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_978 = tensor.insert_slice %collapsed_977 into %inserted_slice_968[384] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_979 = tensor.extract_slice %768#0[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_980 = tensor.extract_slice %extracted_slice_889[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %840 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_980, %extracted_slice_979 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %841 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%840 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %842 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %841) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %843 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%842 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_981 = tensor.extract %843[] : tensor<f32>
    %844 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%842, %extracted_981 : tensor<1024xf32>, f32) outs(%842 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %845 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%844 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_982 = tensor.extract %845[] : tensor<f32>
    %expanded_983 = tensor.expand_shape %844 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_984 = tensor.extract_slice %extracted_slice_890[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_985 = tensor.extract_slice %inserted_slice_978[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_986 = tensor.reshape %extracted_slice_985(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %846 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_986 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %847 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_983, %extracted_982, %extracted_slice_984 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%846 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_987 = tensor.collapse_shape %847 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_988 = tensor.insert_slice %collapsed_987 into %inserted_slice_978[432] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_989 = tensor.extract_slice %768#0[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_990 = tensor.extract_slice %extracted_slice_889[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %848 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_990, %extracted_slice_989 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %849 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%848 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %850 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %849) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %851 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%850 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_991 = tensor.extract %851[] : tensor<f32>
    %852 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%850, %extracted_991 : tensor<1024xf32>, f32) outs(%850 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %853 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%852 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_992 = tensor.extract %853[] : tensor<f32>
    %expanded_993 = tensor.expand_shape %852 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_994 = tensor.extract_slice %extracted_slice_890[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_995 = tensor.extract_slice %inserted_slice_988[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_996 = tensor.reshape %extracted_slice_995(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %854 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_996 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %855 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_993, %extracted_992, %extracted_slice_994 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%854 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_997 = tensor.collapse_shape %855 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_998 = tensor.insert_slice %collapsed_997 into %inserted_slice_988[480] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_999 = tensor.extract_slice %768#0[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_1000 = tensor.extract_slice %extracted_slice_889[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %856 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_1000, %extracted_slice_999 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %857 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%856 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %858 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %857) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %859 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%858 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_1001 = tensor.extract %859[] : tensor<f32>
    %860 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%858, %extracted_1001 : tensor<1024xf32>, f32) outs(%858 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %861 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%860 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_1002 = tensor.extract %861[] : tensor<f32>
    %expanded_1003 = tensor.expand_shape %860 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_1004 = tensor.extract_slice %extracted_slice_890[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_1005 = tensor.extract_slice %inserted_slice_998[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_1006 = tensor.reshape %extracted_slice_1005(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %862 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_1006 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %863 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_1003, %extracted_1002, %extracted_slice_1004 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%862 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_1007 = tensor.collapse_shape %863 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_1008 = tensor.insert_slice %collapsed_1007 into %inserted_slice_998[528] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_1009 = tensor.extract_slice %768#0[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_1010 = tensor.extract_slice %extracted_slice_889[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %864 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_1010, %extracted_slice_1009 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %865 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%864 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %866 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %865) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %867 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%866 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_1011 = tensor.extract %867[] : tensor<f32>
    %868 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%866, %extracted_1011 : tensor<1024xf32>, f32) outs(%866 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %869 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%868 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_1012 = tensor.extract %869[] : tensor<f32>
    %expanded_1013 = tensor.expand_shape %868 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_1014 = tensor.extract_slice %extracted_slice_890[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_1015 = tensor.extract_slice %inserted_slice_1008[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_1016 = tensor.reshape %extracted_slice_1015(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %870 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_1016 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %871 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_1013, %extracted_1012, %extracted_slice_1014 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%870 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_1017 = tensor.collapse_shape %871 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_1018 = tensor.insert_slice %collapsed_1017 into %inserted_slice_1008[576] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_1019 = tensor.extract_slice %768#0[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_1020 = tensor.extract_slice %extracted_slice_889[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %872 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_1020, %extracted_slice_1019 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %873 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%872 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %874 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %873) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %875 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%874 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_1021 = tensor.extract %875[] : tensor<f32>
    %876 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%874, %extracted_1021 : tensor<1024xf32>, f32) outs(%874 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %877 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%876 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_1022 = tensor.extract %877[] : tensor<f32>
    %expanded_1023 = tensor.expand_shape %876 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_1024 = tensor.extract_slice %extracted_slice_890[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_1025 = tensor.extract_slice %inserted_slice_1018[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_1026 = tensor.reshape %extracted_slice_1025(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %878 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_1026 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %879 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_1023, %extracted_1022, %extracted_slice_1024 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%878 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_1027 = tensor.collapse_shape %879 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_1028 = tensor.insert_slice %collapsed_1027 into %inserted_slice_1018[624] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_1029 = tensor.extract_slice %768#0[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_1030 = tensor.extract_slice %extracted_slice_889[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %880 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_1030, %extracted_slice_1029 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %881 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%880 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %882 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %881) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %883 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%882 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_1031 = tensor.extract %883[] : tensor<f32>
    %884 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%882, %extracted_1031 : tensor<1024xf32>, f32) outs(%882 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %885 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%884 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_1032 = tensor.extract %885[] : tensor<f32>
    %expanded_1033 = tensor.expand_shape %884 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_1034 = tensor.extract_slice %extracted_slice_890[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_1035 = tensor.extract_slice %inserted_slice_1028[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_1036 = tensor.reshape %extracted_slice_1035(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %886 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_1036 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %887 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_1033, %extracted_1032, %extracted_slice_1034 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%886 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_1037 = tensor.collapse_shape %887 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_1038 = tensor.insert_slice %collapsed_1037 into %inserted_slice_1028[672] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_1039 = tensor.extract_slice %768#0[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_1040 = tensor.extract_slice %extracted_slice_889[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %888 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_1040, %extracted_slice_1039 : tensor<1024x48xf32>, tensor<48xf32>) outs(%19 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %889 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%888 : tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.divf %in, %cst_0 : f32
      linalg.yield %916 : f32
    } -> tensor<1024xf32>
    %890 = scf.for %arg16 = %17 to %c1024 step %c1 iter_args(%arg17 = %889) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg17[%arg16] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %891 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%890 : tensor<1024xf32>) outs(%23 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.maxnumf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_1041 = tensor.extract %891[] : tensor<f32>
    %892 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%890, %extracted_1041 : tensor<1024xf32>, f32) outs(%890 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.subf %in, %in_1058 : f32
      %917 = math.exp %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1024xf32>
    %893 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%892 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.addf %in, %out : f32
      linalg.yield %916 : f32
    } -> tensor<f32>
    %extracted_1042 = tensor.extract %893[] : tensor<f32>
    %expanded_1043 = tensor.expand_shape %892 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_1044 = tensor.extract_slice %extracted_slice_890[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_1045 = tensor.extract_slice %inserted_slice_1038[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_1046 = tensor.reshape %extracted_slice_1045(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %894 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_1046 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %895 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_1043, %extracted_1042, %extracted_slice_1044 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%894 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.divf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      %918 = arith.addf %out, %917 : f32
      linalg.yield %918 : f32
    } -> tensor<1x48xf32>
    %collapsed_1047 = tensor.collapse_shape %895 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_1048 = tensor.insert_slice %collapsed_1047 into %inserted_slice_1038[720] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %896 = bufferization.materialize_in_destination %inserted_slice_1048 in %762 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_1049 = tensor.extract_slice %arg9[5, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %897 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_1049, %896 : tensor<768x768xf32>, tensor<768xf32>) outs(%757#1 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<768xf32>
    %extracted_slice_1050 = tensor.extract_slice %arg13[5, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %898 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%897 : tensor<768xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.mulf %in, %in : f32
      %917 = arith.addf %916, %out : f32
      linalg.yield %917 : f32
    } -> tensor<f32>
    %extracted_1051 = tensor.extract %898[] : tensor<f32>
    %899 = arith.divf %extracted_1051, %cst_3 : f32
    %900 = arith.addf %899, %cst_4 : f32
    %901 = math.rsqrt %900 : f32
    %902 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%897, %901, %extracted_slice_1050 : tensor<768xf32>, f32, tensor<768xf32>) outs(%896 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.mulf %916, %in_1059 : f32
      linalg.yield %917 : f32
    } -> tensor<768xf32>
    %903 = bufferization.materialize_in_destination %902 in %896 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_1052 = tensor.extract_slice %arg10[5, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %extracted_slice_1053 = tensor.extract_slice %arg12[5, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %904 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_1052, %903 : tensor<2048x768xf32>, tensor<768xf32>) outs(%158 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<2048xf32>
    %905 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_1053, %903 : tensor<2048x768xf32>, tensor<768xf32>) outs(%158 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_1058: f32, %out: f32):
      %916 = arith.mulf %in, %in_1058 : f32
      %917 = arith.addf %out, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<2048xf32>
    %extracted_slice_1054 = tensor.extract_slice %arg11[5, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
    %906:2 = linalg.generic {indexing_maps = [#map3, #map4, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_1054, %905 : tensor<768x2048xf32>, tensor<2048xf32>) outs(%904, %897 : tensor<2048xf32>, tensor<768xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %out: f32, %out_1059: f32):
      %916 = arith.negf %out : f32
      %917 = math.exp %916 : f32
      %918 = arith.addf %917, %cst_5 : f32
      %919 = arith.divf %cst_5, %918 : f32
      %920 = arith.mulf %out, %919 : f32
      %921 = arith.mulf %920, %in_1058 : f32
      %922 = arith.mulf %in, %921 : f32
      %923 = arith.addf %out_1059, %922 : f32
      linalg.yield %921, %923 : f32, f32
    } -> (tensor<2048xf32>, tensor<768xf32>)
    %907 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%906#1 : tensor<768xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %916 = arith.mulf %in, %in : f32
      %917 = arith.addf %916, %out : f32
      linalg.yield %917 : f32
    } -> tensor<f32>
    %extracted_1055 = tensor.extract %907[] : tensor<f32>
    %908 = arith.divf %extracted_1055, %cst_3 : f32
    %909 = arith.addf %908, %cst_4 : f32
    %910 = math.rsqrt %909 : f32
    %911 = tensor.empty() : tensor<34048x768xf32>
    %912 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%911 : tensor<34048x768xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<34048x768xf32>
    %inserted_slice_1056 = tensor.insert_slice %arg15 into %912[0, 0] [32000, 768] [1, 1] : tensor<32000x768xf32> into tensor<34048x768xf32>
    %913 = tensor.empty() : tensor<34048xf32>
    %914 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%913 : tensor<34048xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<34048xf32>
    %915 = linalg.generic {indexing_maps = [#map3, #map4, #map10, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%inserted_slice_1056, %906#1, %910, %arg14 : tensor<34048x768xf32>, tensor<768xf32>, f32, tensor<768xf32>) outs(%914 : tensor<34048xf32>) {
    ^bb0(%in: f32, %in_1058: f32, %in_1059: f32, %in_1060: f32, %out: f32):
      %916 = arith.mulf %in_1058, %in_1059 : f32
      %917 = arith.mulf %916, %in_1060 : f32
      %918 = arith.mulf %in, %917 : f32
      %919 = arith.addf %out, %918 : f32
      linalg.yield %919 : f32
    } -> tensor<34048xf32>
    %extracted_slice_1057 = tensor.extract_slice %915[0] [32000] [1] : tensor<34048xf32> to tensor<32000xf32>
    return %extracted_slice_1057 : tensor<32000xf32>
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
    %cst = arith.constant 0xFFC00000 : f32
    %cst_0 = arith.constant 6.92820311 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %cst_2 = arith.constant 0xFF800000 : f32
    %0 = arith.addi %arg3, %c1 : index
    %1 = tensor.empty() : tensor<768xf32>
    %extracted_slice = tensor.extract_slice %arg0[0] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_3 = tensor.extract_slice %arg1[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %2 = tensor.empty() : tensor<1024xf32>
    %3 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%2 : tensor<1024xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1024xf32>
    %4 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_3, %extracted_slice : tensor<1024x48xf32>, tensor<48xf32>) outs(%3 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_157: f32, %out: f32):
      %135 = arith.mulf %in, %in_157 : f32
      %136 = arith.addf %out, %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %5 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%4 : tensor<1024xf32>) outs(%2 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.divf %in, %cst_0 : f32
      linalg.yield %135 : f32
    } -> tensor<1024xf32>
    %6 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %5) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %7 = tensor.empty() : tensor<f32>
    %8 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%7 : tensor<f32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    } -> tensor<f32>
    %9 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%6 : tensor<1024xf32>) outs(%8 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.maxnumf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted = tensor.extract %9[] : tensor<f32>
    %10 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%6, %extracted : tensor<1024xf32>, f32) outs(%6 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_157: f32, %out: f32):
      %135 = arith.subf %in, %in_157 : f32
      %136 = math.exp %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %11 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%7 : tensor<f32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<f32>
    %12 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%10 : tensor<1024xf32>) outs(%11 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.addf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_4 = tensor.extract %12[] : tensor<f32>
    %expanded = tensor.expand_shape %10 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_5 = tensor.extract_slice %arg2[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_6 = tensor.extract_slice %1[0] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %c48 = arith.constant 48 : index
    %from_elements = tensor.from_elements %c1, %c48 : tensor<2xindex>
    %reshape = tensor.reshape %extracted_slice_6(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %13 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %14 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded, %extracted_4, %extracted_slice_5 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%13 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_157: f32, %in_158: f32, %out: f32):
      %135 = arith.divf %in, %in_157 : f32
      %136 = arith.mulf %135, %in_158 : f32
      %137 = arith.addf %out, %136 : f32
      linalg.yield %137 : f32
    } -> tensor<1x48xf32>
    %collapsed = tensor.collapse_shape %14 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice = tensor.insert_slice %collapsed into %1[0] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_7 = tensor.extract_slice %arg0[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_8 = tensor.extract_slice %arg1[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %15 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_8, %extracted_slice_7 : tensor<1024x48xf32>, tensor<48xf32>) outs(%3 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_157: f32, %out: f32):
      %135 = arith.mulf %in, %in_157 : f32
      %136 = arith.addf %out, %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %16 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%15 : tensor<1024xf32>) outs(%2 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.divf %in, %cst_0 : f32
      linalg.yield %135 : f32
    } -> tensor<1024xf32>
    %17 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %16) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %18 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%17 : tensor<1024xf32>) outs(%8 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.maxnumf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_9 = tensor.extract %18[] : tensor<f32>
    %19 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%17, %extracted_9 : tensor<1024xf32>, f32) outs(%17 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_157: f32, %out: f32):
      %135 = arith.subf %in, %in_157 : f32
      %136 = math.exp %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %20 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%19 : tensor<1024xf32>) outs(%11 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.addf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_10 = tensor.extract %20[] : tensor<f32>
    %expanded_11 = tensor.expand_shape %19 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_12 = tensor.extract_slice %arg2[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_13 = tensor.extract_slice %inserted_slice[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_14 = tensor.reshape %extracted_slice_13(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %21 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_14 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %22 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_11, %extracted_10, %extracted_slice_12 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%21 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_157: f32, %in_158: f32, %out: f32):
      %135 = arith.divf %in, %in_157 : f32
      %136 = arith.mulf %135, %in_158 : f32
      %137 = arith.addf %out, %136 : f32
      linalg.yield %137 : f32
    } -> tensor<1x48xf32>
    %collapsed_15 = tensor.collapse_shape %22 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_16 = tensor.insert_slice %collapsed_15 into %inserted_slice[48] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_17 = tensor.extract_slice %arg0[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_18 = tensor.extract_slice %arg1[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %23 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_18, %extracted_slice_17 : tensor<1024x48xf32>, tensor<48xf32>) outs(%3 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_157: f32, %out: f32):
      %135 = arith.mulf %in, %in_157 : f32
      %136 = arith.addf %out, %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %24 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%23 : tensor<1024xf32>) outs(%2 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.divf %in, %cst_0 : f32
      linalg.yield %135 : f32
    } -> tensor<1024xf32>
    %25 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %24) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %26 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%25 : tensor<1024xf32>) outs(%8 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.maxnumf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_19 = tensor.extract %26[] : tensor<f32>
    %27 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%25, %extracted_19 : tensor<1024xf32>, f32) outs(%25 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_157: f32, %out: f32):
      %135 = arith.subf %in, %in_157 : f32
      %136 = math.exp %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %28 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%27 : tensor<1024xf32>) outs(%11 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.addf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_20 = tensor.extract %28[] : tensor<f32>
    %expanded_21 = tensor.expand_shape %27 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_22 = tensor.extract_slice %arg2[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_23 = tensor.extract_slice %inserted_slice_16[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_24 = tensor.reshape %extracted_slice_23(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %29 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_24 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %30 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_21, %extracted_20, %extracted_slice_22 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%29 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_157: f32, %in_158: f32, %out: f32):
      %135 = arith.divf %in, %in_157 : f32
      %136 = arith.mulf %135, %in_158 : f32
      %137 = arith.addf %out, %136 : f32
      linalg.yield %137 : f32
    } -> tensor<1x48xf32>
    %collapsed_25 = tensor.collapse_shape %30 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_26 = tensor.insert_slice %collapsed_25 into %inserted_slice_16[96] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_27 = tensor.extract_slice %arg0[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_28 = tensor.extract_slice %arg1[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %31 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_28, %extracted_slice_27 : tensor<1024x48xf32>, tensor<48xf32>) outs(%3 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_157: f32, %out: f32):
      %135 = arith.mulf %in, %in_157 : f32
      %136 = arith.addf %out, %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %32 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%31 : tensor<1024xf32>) outs(%2 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.divf %in, %cst_0 : f32
      linalg.yield %135 : f32
    } -> tensor<1024xf32>
    %33 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %32) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %34 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%33 : tensor<1024xf32>) outs(%8 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.maxnumf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_29 = tensor.extract %34[] : tensor<f32>
    %35 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%33, %extracted_29 : tensor<1024xf32>, f32) outs(%33 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_157: f32, %out: f32):
      %135 = arith.subf %in, %in_157 : f32
      %136 = math.exp %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %36 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%35 : tensor<1024xf32>) outs(%11 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.addf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_30 = tensor.extract %36[] : tensor<f32>
    %expanded_31 = tensor.expand_shape %35 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_32 = tensor.extract_slice %arg2[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_33 = tensor.extract_slice %inserted_slice_26[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_34 = tensor.reshape %extracted_slice_33(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %37 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_34 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %38 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_31, %extracted_30, %extracted_slice_32 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%37 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_157: f32, %in_158: f32, %out: f32):
      %135 = arith.divf %in, %in_157 : f32
      %136 = arith.mulf %135, %in_158 : f32
      %137 = arith.addf %out, %136 : f32
      linalg.yield %137 : f32
    } -> tensor<1x48xf32>
    %collapsed_35 = tensor.collapse_shape %38 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_36 = tensor.insert_slice %collapsed_35 into %inserted_slice_26[144] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_37 = tensor.extract_slice %arg0[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_38 = tensor.extract_slice %arg1[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %39 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_38, %extracted_slice_37 : tensor<1024x48xf32>, tensor<48xf32>) outs(%3 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_157: f32, %out: f32):
      %135 = arith.mulf %in, %in_157 : f32
      %136 = arith.addf %out, %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %40 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%39 : tensor<1024xf32>) outs(%2 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.divf %in, %cst_0 : f32
      linalg.yield %135 : f32
    } -> tensor<1024xf32>
    %41 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %40) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %42 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%41 : tensor<1024xf32>) outs(%8 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.maxnumf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_39 = tensor.extract %42[] : tensor<f32>
    %43 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%41, %extracted_39 : tensor<1024xf32>, f32) outs(%41 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_157: f32, %out: f32):
      %135 = arith.subf %in, %in_157 : f32
      %136 = math.exp %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %44 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%43 : tensor<1024xf32>) outs(%11 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.addf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_40 = tensor.extract %44[] : tensor<f32>
    %expanded_41 = tensor.expand_shape %43 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_42 = tensor.extract_slice %arg2[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_43 = tensor.extract_slice %inserted_slice_36[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_44 = tensor.reshape %extracted_slice_43(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %45 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_44 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %46 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_41, %extracted_40, %extracted_slice_42 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%45 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_157: f32, %in_158: f32, %out: f32):
      %135 = arith.divf %in, %in_157 : f32
      %136 = arith.mulf %135, %in_158 : f32
      %137 = arith.addf %out, %136 : f32
      linalg.yield %137 : f32
    } -> tensor<1x48xf32>
    %collapsed_45 = tensor.collapse_shape %46 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_46 = tensor.insert_slice %collapsed_45 into %inserted_slice_36[192] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_47 = tensor.extract_slice %arg0[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_48 = tensor.extract_slice %arg1[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %47 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_48, %extracted_slice_47 : tensor<1024x48xf32>, tensor<48xf32>) outs(%3 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_157: f32, %out: f32):
      %135 = arith.mulf %in, %in_157 : f32
      %136 = arith.addf %out, %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %48 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%47 : tensor<1024xf32>) outs(%2 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.divf %in, %cst_0 : f32
      linalg.yield %135 : f32
    } -> tensor<1024xf32>
    %49 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %48) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %50 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%49 : tensor<1024xf32>) outs(%8 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.maxnumf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_49 = tensor.extract %50[] : tensor<f32>
    %51 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%49, %extracted_49 : tensor<1024xf32>, f32) outs(%49 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_157: f32, %out: f32):
      %135 = arith.subf %in, %in_157 : f32
      %136 = math.exp %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %52 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%51 : tensor<1024xf32>) outs(%11 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.addf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_50 = tensor.extract %52[] : tensor<f32>
    %expanded_51 = tensor.expand_shape %51 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_52 = tensor.extract_slice %arg2[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_53 = tensor.extract_slice %inserted_slice_46[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_54 = tensor.reshape %extracted_slice_53(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %53 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_54 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %54 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_51, %extracted_50, %extracted_slice_52 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%53 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_157: f32, %in_158: f32, %out: f32):
      %135 = arith.divf %in, %in_157 : f32
      %136 = arith.mulf %135, %in_158 : f32
      %137 = arith.addf %out, %136 : f32
      linalg.yield %137 : f32
    } -> tensor<1x48xf32>
    %collapsed_55 = tensor.collapse_shape %54 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_56 = tensor.insert_slice %collapsed_55 into %inserted_slice_46[240] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_57 = tensor.extract_slice %arg0[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_58 = tensor.extract_slice %arg1[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %55 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_58, %extracted_slice_57 : tensor<1024x48xf32>, tensor<48xf32>) outs(%3 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_157: f32, %out: f32):
      %135 = arith.mulf %in, %in_157 : f32
      %136 = arith.addf %out, %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %56 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%55 : tensor<1024xf32>) outs(%2 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.divf %in, %cst_0 : f32
      linalg.yield %135 : f32
    } -> tensor<1024xf32>
    %57 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %56) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %58 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%57 : tensor<1024xf32>) outs(%8 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.maxnumf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_59 = tensor.extract %58[] : tensor<f32>
    %59 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%57, %extracted_59 : tensor<1024xf32>, f32) outs(%57 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_157: f32, %out: f32):
      %135 = arith.subf %in, %in_157 : f32
      %136 = math.exp %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %60 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%59 : tensor<1024xf32>) outs(%11 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.addf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_60 = tensor.extract %60[] : tensor<f32>
    %expanded_61 = tensor.expand_shape %59 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_62 = tensor.extract_slice %arg2[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_63 = tensor.extract_slice %inserted_slice_56[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_64 = tensor.reshape %extracted_slice_63(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %61 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_64 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %62 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_61, %extracted_60, %extracted_slice_62 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%61 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_157: f32, %in_158: f32, %out: f32):
      %135 = arith.divf %in, %in_157 : f32
      %136 = arith.mulf %135, %in_158 : f32
      %137 = arith.addf %out, %136 : f32
      linalg.yield %137 : f32
    } -> tensor<1x48xf32>
    %collapsed_65 = tensor.collapse_shape %62 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_66 = tensor.insert_slice %collapsed_65 into %inserted_slice_56[288] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_67 = tensor.extract_slice %arg0[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_68 = tensor.extract_slice %arg1[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %63 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_68, %extracted_slice_67 : tensor<1024x48xf32>, tensor<48xf32>) outs(%3 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_157: f32, %out: f32):
      %135 = arith.mulf %in, %in_157 : f32
      %136 = arith.addf %out, %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %64 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%63 : tensor<1024xf32>) outs(%2 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.divf %in, %cst_0 : f32
      linalg.yield %135 : f32
    } -> tensor<1024xf32>
    %65 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %64) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %66 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%65 : tensor<1024xf32>) outs(%8 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.maxnumf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_69 = tensor.extract %66[] : tensor<f32>
    %67 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%65, %extracted_69 : tensor<1024xf32>, f32) outs(%65 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_157: f32, %out: f32):
      %135 = arith.subf %in, %in_157 : f32
      %136 = math.exp %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %68 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%67 : tensor<1024xf32>) outs(%11 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.addf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_70 = tensor.extract %68[] : tensor<f32>
    %expanded_71 = tensor.expand_shape %67 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_72 = tensor.extract_slice %arg2[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_73 = tensor.extract_slice %inserted_slice_66[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_74 = tensor.reshape %extracted_slice_73(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %69 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_74 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %70 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_71, %extracted_70, %extracted_slice_72 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%69 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_157: f32, %in_158: f32, %out: f32):
      %135 = arith.divf %in, %in_157 : f32
      %136 = arith.mulf %135, %in_158 : f32
      %137 = arith.addf %out, %136 : f32
      linalg.yield %137 : f32
    } -> tensor<1x48xf32>
    %collapsed_75 = tensor.collapse_shape %70 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_76 = tensor.insert_slice %collapsed_75 into %inserted_slice_66[336] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_77 = tensor.extract_slice %arg0[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_78 = tensor.extract_slice %arg1[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %71 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_78, %extracted_slice_77 : tensor<1024x48xf32>, tensor<48xf32>) outs(%3 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_157: f32, %out: f32):
      %135 = arith.mulf %in, %in_157 : f32
      %136 = arith.addf %out, %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %72 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%71 : tensor<1024xf32>) outs(%2 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.divf %in, %cst_0 : f32
      linalg.yield %135 : f32
    } -> tensor<1024xf32>
    %73 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %72) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %74 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%73 : tensor<1024xf32>) outs(%8 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.maxnumf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_79 = tensor.extract %74[] : tensor<f32>
    %75 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%73, %extracted_79 : tensor<1024xf32>, f32) outs(%73 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_157: f32, %out: f32):
      %135 = arith.subf %in, %in_157 : f32
      %136 = math.exp %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %76 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%75 : tensor<1024xf32>) outs(%11 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.addf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_80 = tensor.extract %76[] : tensor<f32>
    %expanded_81 = tensor.expand_shape %75 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_82 = tensor.extract_slice %arg2[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_83 = tensor.extract_slice %inserted_slice_76[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_84 = tensor.reshape %extracted_slice_83(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %77 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_84 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %78 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_81, %extracted_80, %extracted_slice_82 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%77 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_157: f32, %in_158: f32, %out: f32):
      %135 = arith.divf %in, %in_157 : f32
      %136 = arith.mulf %135, %in_158 : f32
      %137 = arith.addf %out, %136 : f32
      linalg.yield %137 : f32
    } -> tensor<1x48xf32>
    %collapsed_85 = tensor.collapse_shape %78 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_86 = tensor.insert_slice %collapsed_85 into %inserted_slice_76[384] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_87 = tensor.extract_slice %arg0[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_88 = tensor.extract_slice %arg1[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %79 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_88, %extracted_slice_87 : tensor<1024x48xf32>, tensor<48xf32>) outs(%3 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_157: f32, %out: f32):
      %135 = arith.mulf %in, %in_157 : f32
      %136 = arith.addf %out, %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %80 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%79 : tensor<1024xf32>) outs(%2 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.divf %in, %cst_0 : f32
      linalg.yield %135 : f32
    } -> tensor<1024xf32>
    %81 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %80) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %82 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%81 : tensor<1024xf32>) outs(%8 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.maxnumf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_89 = tensor.extract %82[] : tensor<f32>
    %83 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%81, %extracted_89 : tensor<1024xf32>, f32) outs(%81 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_157: f32, %out: f32):
      %135 = arith.subf %in, %in_157 : f32
      %136 = math.exp %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %84 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%83 : tensor<1024xf32>) outs(%11 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.addf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_90 = tensor.extract %84[] : tensor<f32>
    %expanded_91 = tensor.expand_shape %83 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_92 = tensor.extract_slice %arg2[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_93 = tensor.extract_slice %inserted_slice_86[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_94 = tensor.reshape %extracted_slice_93(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %85 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_94 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %86 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_91, %extracted_90, %extracted_slice_92 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%85 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_157: f32, %in_158: f32, %out: f32):
      %135 = arith.divf %in, %in_157 : f32
      %136 = arith.mulf %135, %in_158 : f32
      %137 = arith.addf %out, %136 : f32
      linalg.yield %137 : f32
    } -> tensor<1x48xf32>
    %collapsed_95 = tensor.collapse_shape %86 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_96 = tensor.insert_slice %collapsed_95 into %inserted_slice_86[432] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_97 = tensor.extract_slice %arg0[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_98 = tensor.extract_slice %arg1[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %87 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_98, %extracted_slice_97 : tensor<1024x48xf32>, tensor<48xf32>) outs(%3 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_157: f32, %out: f32):
      %135 = arith.mulf %in, %in_157 : f32
      %136 = arith.addf %out, %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %88 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%87 : tensor<1024xf32>) outs(%2 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.divf %in, %cst_0 : f32
      linalg.yield %135 : f32
    } -> tensor<1024xf32>
    %89 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %88) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %90 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%89 : tensor<1024xf32>) outs(%8 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.maxnumf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_99 = tensor.extract %90[] : tensor<f32>
    %91 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%89, %extracted_99 : tensor<1024xf32>, f32) outs(%89 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_157: f32, %out: f32):
      %135 = arith.subf %in, %in_157 : f32
      %136 = math.exp %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %92 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%91 : tensor<1024xf32>) outs(%11 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.addf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_100 = tensor.extract %92[] : tensor<f32>
    %expanded_101 = tensor.expand_shape %91 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_102 = tensor.extract_slice %arg2[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_103 = tensor.extract_slice %inserted_slice_96[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_104 = tensor.reshape %extracted_slice_103(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %93 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_104 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %94 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_101, %extracted_100, %extracted_slice_102 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%93 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_157: f32, %in_158: f32, %out: f32):
      %135 = arith.divf %in, %in_157 : f32
      %136 = arith.mulf %135, %in_158 : f32
      %137 = arith.addf %out, %136 : f32
      linalg.yield %137 : f32
    } -> tensor<1x48xf32>
    %collapsed_105 = tensor.collapse_shape %94 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_106 = tensor.insert_slice %collapsed_105 into %inserted_slice_96[480] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_107 = tensor.extract_slice %arg0[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_108 = tensor.extract_slice %arg1[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %95 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_108, %extracted_slice_107 : tensor<1024x48xf32>, tensor<48xf32>) outs(%3 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_157: f32, %out: f32):
      %135 = arith.mulf %in, %in_157 : f32
      %136 = arith.addf %out, %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %96 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%95 : tensor<1024xf32>) outs(%2 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.divf %in, %cst_0 : f32
      linalg.yield %135 : f32
    } -> tensor<1024xf32>
    %97 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %96) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %98 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%97 : tensor<1024xf32>) outs(%8 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.maxnumf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_109 = tensor.extract %98[] : tensor<f32>
    %99 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%97, %extracted_109 : tensor<1024xf32>, f32) outs(%97 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_157: f32, %out: f32):
      %135 = arith.subf %in, %in_157 : f32
      %136 = math.exp %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %100 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%99 : tensor<1024xf32>) outs(%11 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.addf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_110 = tensor.extract %100[] : tensor<f32>
    %expanded_111 = tensor.expand_shape %99 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_112 = tensor.extract_slice %arg2[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_113 = tensor.extract_slice %inserted_slice_106[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_114 = tensor.reshape %extracted_slice_113(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %101 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_114 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %102 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_111, %extracted_110, %extracted_slice_112 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%101 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_157: f32, %in_158: f32, %out: f32):
      %135 = arith.divf %in, %in_157 : f32
      %136 = arith.mulf %135, %in_158 : f32
      %137 = arith.addf %out, %136 : f32
      linalg.yield %137 : f32
    } -> tensor<1x48xf32>
    %collapsed_115 = tensor.collapse_shape %102 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_116 = tensor.insert_slice %collapsed_115 into %inserted_slice_106[528] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_117 = tensor.extract_slice %arg0[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_118 = tensor.extract_slice %arg1[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %103 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_118, %extracted_slice_117 : tensor<1024x48xf32>, tensor<48xf32>) outs(%3 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_157: f32, %out: f32):
      %135 = arith.mulf %in, %in_157 : f32
      %136 = arith.addf %out, %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %104 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%103 : tensor<1024xf32>) outs(%2 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.divf %in, %cst_0 : f32
      linalg.yield %135 : f32
    } -> tensor<1024xf32>
    %105 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %104) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %106 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%105 : tensor<1024xf32>) outs(%8 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.maxnumf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_119 = tensor.extract %106[] : tensor<f32>
    %107 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%105, %extracted_119 : tensor<1024xf32>, f32) outs(%105 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_157: f32, %out: f32):
      %135 = arith.subf %in, %in_157 : f32
      %136 = math.exp %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %108 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%107 : tensor<1024xf32>) outs(%11 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.addf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_120 = tensor.extract %108[] : tensor<f32>
    %expanded_121 = tensor.expand_shape %107 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_122 = tensor.extract_slice %arg2[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_123 = tensor.extract_slice %inserted_slice_116[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_124 = tensor.reshape %extracted_slice_123(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %109 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_124 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %110 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_121, %extracted_120, %extracted_slice_122 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%109 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_157: f32, %in_158: f32, %out: f32):
      %135 = arith.divf %in, %in_157 : f32
      %136 = arith.mulf %135, %in_158 : f32
      %137 = arith.addf %out, %136 : f32
      linalg.yield %137 : f32
    } -> tensor<1x48xf32>
    %collapsed_125 = tensor.collapse_shape %110 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_126 = tensor.insert_slice %collapsed_125 into %inserted_slice_116[576] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_127 = tensor.extract_slice %arg0[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_128 = tensor.extract_slice %arg1[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %111 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_128, %extracted_slice_127 : tensor<1024x48xf32>, tensor<48xf32>) outs(%3 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_157: f32, %out: f32):
      %135 = arith.mulf %in, %in_157 : f32
      %136 = arith.addf %out, %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %112 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%111 : tensor<1024xf32>) outs(%2 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.divf %in, %cst_0 : f32
      linalg.yield %135 : f32
    } -> tensor<1024xf32>
    %113 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %112) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %114 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%113 : tensor<1024xf32>) outs(%8 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.maxnumf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_129 = tensor.extract %114[] : tensor<f32>
    %115 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%113, %extracted_129 : tensor<1024xf32>, f32) outs(%113 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_157: f32, %out: f32):
      %135 = arith.subf %in, %in_157 : f32
      %136 = math.exp %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %116 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%115 : tensor<1024xf32>) outs(%11 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.addf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_130 = tensor.extract %116[] : tensor<f32>
    %expanded_131 = tensor.expand_shape %115 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_132 = tensor.extract_slice %arg2[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_133 = tensor.extract_slice %inserted_slice_126[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_134 = tensor.reshape %extracted_slice_133(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %117 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_134 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %118 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_131, %extracted_130, %extracted_slice_132 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%117 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_157: f32, %in_158: f32, %out: f32):
      %135 = arith.divf %in, %in_157 : f32
      %136 = arith.mulf %135, %in_158 : f32
      %137 = arith.addf %out, %136 : f32
      linalg.yield %137 : f32
    } -> tensor<1x48xf32>
    %collapsed_135 = tensor.collapse_shape %118 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_136 = tensor.insert_slice %collapsed_135 into %inserted_slice_126[624] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_137 = tensor.extract_slice %arg0[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_138 = tensor.extract_slice %arg1[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %119 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_138, %extracted_slice_137 : tensor<1024x48xf32>, tensor<48xf32>) outs(%3 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_157: f32, %out: f32):
      %135 = arith.mulf %in, %in_157 : f32
      %136 = arith.addf %out, %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %120 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%119 : tensor<1024xf32>) outs(%2 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.divf %in, %cst_0 : f32
      linalg.yield %135 : f32
    } -> tensor<1024xf32>
    %121 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %120) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %122 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%121 : tensor<1024xf32>) outs(%8 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.maxnumf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_139 = tensor.extract %122[] : tensor<f32>
    %123 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%121, %extracted_139 : tensor<1024xf32>, f32) outs(%121 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_157: f32, %out: f32):
      %135 = arith.subf %in, %in_157 : f32
      %136 = math.exp %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %124 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%123 : tensor<1024xf32>) outs(%11 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.addf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_140 = tensor.extract %124[] : tensor<f32>
    %expanded_141 = tensor.expand_shape %123 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_142 = tensor.extract_slice %arg2[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_143 = tensor.extract_slice %inserted_slice_136[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_144 = tensor.reshape %extracted_slice_143(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %125 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_144 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %126 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_141, %extracted_140, %extracted_slice_142 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%125 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_157: f32, %in_158: f32, %out: f32):
      %135 = arith.divf %in, %in_157 : f32
      %136 = arith.mulf %135, %in_158 : f32
      %137 = arith.addf %out, %136 : f32
      linalg.yield %137 : f32
    } -> tensor<1x48xf32>
    %collapsed_145 = tensor.collapse_shape %126 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_146 = tensor.insert_slice %collapsed_145 into %inserted_slice_136[672] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_147 = tensor.extract_slice %arg0[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_148 = tensor.extract_slice %arg1[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %127 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_148, %extracted_slice_147 : tensor<1024x48xf32>, tensor<48xf32>) outs(%3 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_157: f32, %out: f32):
      %135 = arith.mulf %in, %in_157 : f32
      %136 = arith.addf %out, %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %128 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%127 : tensor<1024xf32>) outs(%2 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.divf %in, %cst_0 : f32
      linalg.yield %135 : f32
    } -> tensor<1024xf32>
    %129 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %128) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_2 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %130 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%129 : tensor<1024xf32>) outs(%8 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.maxnumf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_149 = tensor.extract %130[] : tensor<f32>
    %131 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%129, %extracted_149 : tensor<1024xf32>, f32) outs(%129 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_157: f32, %out: f32):
      %135 = arith.subf %in, %in_157 : f32
      %136 = math.exp %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %132 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%131 : tensor<1024xf32>) outs(%11 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.addf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_150 = tensor.extract %132[] : tensor<f32>
    %expanded_151 = tensor.expand_shape %131 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_152 = tensor.extract_slice %arg2[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_153 = tensor.extract_slice %inserted_slice_146[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_154 = tensor.reshape %extracted_slice_153(%from_elements) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %133 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_154 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    } -> tensor<1x48xf32>
    %134 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_151, %extracted_150, %extracted_slice_152 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%133 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_157: f32, %in_158: f32, %out: f32):
      %135 = arith.divf %in, %in_157 : f32
      %136 = arith.mulf %135, %in_158 : f32
      %137 = arith.addf %out, %136 : f32
      linalg.yield %137 : f32
    } -> tensor<1x48xf32>
    %collapsed_155 = tensor.collapse_shape %134 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_156 = tensor.insert_slice %collapsed_155 into %inserted_slice_146[720] [48] [1] : tensor<48xf32> into tensor<768xf32>
    return %inserted_slice_156 : tensor<768xf32>
  }
  func.func @rmsnorm(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>) -> tensor<768xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 9.99999974E-6 : f32
    %cst_1 = arith.constant 7.680000e+02 : f32
    %0 = tensor.empty() : tensor<f32>
    %1 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    } -> tensor<f32>
    %2 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%arg0 : tensor<768xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %8 = arith.mulf %in, %in : f32
      %9 = arith.addf %8, %out : f32
      linalg.yield %9 : f32
    } -> tensor<f32>
    %extracted = tensor.extract %2[] : tensor<f32>
    %3 = arith.divf %extracted, %cst_1 : f32
    %4 = arith.addf %3, %cst_0 : f32
    %5 = math.rsqrt %4 : f32
    %6 = tensor.empty() : tensor<768xf32>
    %7 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%arg0, %5, %arg1 : tensor<768xf32>, f32, tensor<768xf32>) outs(%6 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_2: f32, %in_3: f32, %out: f32):
      %8 = arith.mulf %in, %in_2 : f32
      %9 = arith.mulf %8, %in_3 : f32
      linalg.yield %9 : f32
    } -> tensor<768xf32>
    return %7 : tensor<768xf32>
  }
  func.func @softmax(%arg0: tensor<1024xf32> {bufferization.writable = true}) -> tensor<1024xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFFC00000 : f32
    %0 = tensor.empty() : tensor<f32>
    %1 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_0 : f32
    } -> tensor<f32>
    %2 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%arg0 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %7 = arith.maxnumf %in, %out : f32
      linalg.yield %7 : f32
    } -> tensor<f32>
    %extracted = tensor.extract %2[] : tensor<f32>
    %3 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%arg0, %extracted : tensor<1024xf32>, f32) outs(%arg0 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %7 = arith.subf %in, %in_2 : f32
      %8 = math.exp %7 : f32
      linalg.yield %8 : f32
    } -> tensor<1024xf32>
    %4 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    } -> tensor<f32>
    %5 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%3 : tensor<1024xf32>) outs(%4 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %7 = arith.addf %in, %out : f32
      linalg.yield %7 : f32
    } -> tensor<f32>
    %extracted_1 = tensor.extract %5[] : tensor<f32>
    %6 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%3, %extracted_1 : tensor<1024xf32>, f32) outs(%arg0 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %7 = arith.divf %in, %in_2 : f32
      linalg.yield %7 : f32
    } -> tensor<1024xf32>
    return %6 : tensor<1024xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["func.func"]} attributes {sym_name = "forward"} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.structured.match ops{["scf.for"]} in %0 : (!transform.any_op) -> !transform.any_op
      %2:2 = transform.split_handle %1 {overflow_result = 1 : i64} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.loop.unroll %2#1 {factor = 6 : i64} : !transform.any_op
      %3 = transform.structured.match ops{["func.func"]} attributes {sym_name = "mha"} in %arg0 : (!transform.any_op) -> !transform.any_op
      %4 = transform.structured.match ops{["scf.for"]} in %3 : (!transform.any_op) -> !transform.any_op
      %5:2 = transform.split_handle %4 {overflow_result = 1 : i64} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.loop.unroll %5#1 {factor = 16 : i64} : !transform.any_op
      transform.yield
    }
  }
}
