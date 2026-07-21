#map = affine_map<() -> ()>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> ()>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d1)>
#map5 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @forward(%arg0: index, %arg1: index, %arg2: tensor<6x1024x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, bufferization.writable = true}, %arg3: tensor<6x1024x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, bufferization.writable = true}, %arg4: tensor<32000x768xf32> {bufferization.buffer_layout = #map3}, %arg5: tensor<6x768xf32> {bufferization.buffer_layout = #map3}, %arg6: tensor<6x768x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}, %arg7: tensor<6x768x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}, %arg8: tensor<6x768x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}, %arg9: tensor<6x768x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}, %arg10: tensor<6x2048x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}, %arg11: tensor<6x768x2048xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}, %arg12: tensor<6x2048x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}, %arg13: tensor<6x768xf32> {bufferization.buffer_layout = #map3}, %arg14: tensor<768xf32> {bufferization.buffer_layout = #map1}, %arg15: tensor<32000x768xf32> {bufferization.buffer_layout = #map3}) -> tensor<32000xf32> {
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
    %extracted_slice = tensor.extract_slice %arg4[%arg0, 0] [1, 768] [1, 1] : tensor<32000x768xf32> to tensor<768xf32>
    %extracted_slice_8 = tensor.extract_slice %arg5[0, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %0 = tensor.empty() : tensor<f32>
    %1 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_0 : f32
    } -> tensor<f32>
    %2 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%extracted_slice : tensor<768xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.mulf %in, %in : f32
      %145 = arith.addf %144, %out : f32
      linalg.yield %145 : f32
    } -> tensor<f32>
    %extracted = tensor.extract %2[] : tensor<f32>
    %3 = arith.divf %extracted, %cst_3 : f32
    %4 = arith.addf %3, %cst_4 : f32
    %5 = math.rsqrt %4 : f32
    %extracted_slice_9 = tensor.extract_slice %arg3[0, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %6 = tensor.empty() : tensor<768xf32>
    %7 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%extracted_slice, %extracted_slice_8 : tensor<768xf32>, tensor<768xf32>) outs(%6 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %5 : f32
      %145 = arith.mulf %144, %in_105 : f32
      linalg.yield %145 : f32
    } -> tensor<768xf32>
    %extracted_slice_10 = tensor.extract_slice %arg6[0, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_11 = tensor.extract_slice %arg7[0, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_12 = tensor.extract_slice %arg8[0, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %8 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_9 : tensor<768xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_0 : f32
    } -> tensor<768xf32>
    %9 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_10, %7 : tensor<768x768xf32>, tensor<768xf32>) outs(%8 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %in_105 : f32
      %145 = arith.addf %out, %144 : f32
      linalg.yield %145 : f32
    } -> tensor<768xf32>
    %10 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_11, %7 : tensor<768x768xf32>, tensor<768xf32>) outs(%8 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %in_105 : f32
      %145 = arith.addf %out, %144 : f32
      linalg.yield %145 : f32
    } -> tensor<768xf32>
    %11 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_12, %7 : tensor<768x768xf32>, tensor<768xf32>) outs(%8 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %in_105 : f32
      %145 = arith.addf %out, %144 : f32
      linalg.yield %145 : f32
    } -> tensor<768xf32>
    %inserted_slice = tensor.insert_slice %11 into %arg3[0, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %12 = arith.index_cast %arg1 : index to i64
    %13 = arith.uitofp %12 : i64 to f32
    %14:2 = scf.for %arg16 = %c0 to %c768 step %c2 iter_args(%arg17 = %9, %arg18 = %10) -> (tensor<768xf32>, tensor<768xf32>) {
      %144 = arith.remui %arg16, %c48 : index
      %145 = arith.index_cast %144 : index to i64
      %146 = arith.uitofp %145 : i64 to f32
      %147 = arith.divf %146, %cst_6 : f32
      %148 = math.powf %cst_7, %147 : f32
      %149 = arith.divf %cst_5, %148 : f32
      %150 = arith.mulf %13, %149 : f32
      %151 = math.cos %150 : f32
      %152 = math.sin %150 : f32
      %153 = arith.addi %arg16, %c1 : index
      %extracted_105 = tensor.extract %arg17[%arg16] : tensor<768xf32>
      %extracted_106 = tensor.extract %arg17[%153] : tensor<768xf32>
      %154 = arith.mulf %extracted_105, %151 : f32
      %155 = arith.mulf %extracted_106, %152 : f32
      %156 = arith.subf %154, %155 : f32
      %inserted = tensor.insert %156 into %arg17[%arg16] : tensor<768xf32>
      %157 = arith.mulf %extracted_105, %152 : f32
      %158 = arith.mulf %extracted_106, %151 : f32
      %159 = arith.addf %157, %158 : f32
      %inserted_107 = tensor.insert %159 into %inserted[%153] : tensor<768xf32>
      %160 = bufferization.materialize_in_destination %inserted_107 in %arg17 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      %161 = arith.cmpi ult, %arg16, %c768 : index
      %162 = scf.if %161 -> (tensor<768xf32>) {
        %extracted_108 = tensor.extract %arg18[%arg16] : tensor<768xf32>
        %extracted_109 = tensor.extract %arg18[%153] : tensor<768xf32>
        %163 = arith.mulf %extracted_108, %151 : f32
        %164 = arith.mulf %extracted_109, %152 : f32
        %165 = arith.subf %163, %164 : f32
        %inserted_110 = tensor.insert %165 into %arg18[%arg16] : tensor<768xf32>
        %166 = arith.mulf %extracted_108, %152 : f32
        %167 = arith.mulf %extracted_109, %151 : f32
        %168 = arith.addf %166, %167 : f32
        %inserted_111 = tensor.insert %168 into %inserted_110[%153] : tensor<768xf32>
        %169 = bufferization.materialize_in_destination %inserted_111 in %arg18 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
        scf.yield %169 : tensor<768xf32>
      } else {
        scf.yield %arg18 : tensor<768xf32>
      }
      scf.yield %160, %162 : tensor<768xf32>, tensor<768xf32>
    }
    %inserted_slice_13 = tensor.insert_slice %14#1 into %arg2[0, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %extracted_slice_14 = tensor.extract_slice %inserted_slice_13[0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %extracted_slice_15 = tensor.extract_slice %inserted_slice[0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %15 = arith.addi %arg1, %c1 : index
    %16 = scf.for %arg16 = %c0 to %c6 step %c1 iter_args(%arg17 = %14#0) -> (tensor<768xf32>) {
      %144 = arith.muli %arg16, %c48 : index
      %145 = tensor.empty() : tensor<1024xf32>
      %146 = scf.for %arg18 = %c0 to %15 step %c1 iter_args(%arg19 = %145) -> (tensor<1024xf32>) {
        %extracted_slice_109 = tensor.extract_slice %14#0[%144] [48] [1] : tensor<768xf32> to tensor<48xf32>
        %extracted_slice_110 = tensor.extract_slice %extracted_slice_14[%arg18, %144] [1, 48] [1, 1] : tensor<1024x768xf32> to tensor<48xf32>
        %155 = linalg.generic {indexing_maps = [#map1, #map1, #map2], iterator_types = ["reduction"]} ins(%extracted_slice_109, %extracted_slice_110 : tensor<48xf32>, tensor<48xf32>) outs(%1 : tensor<f32>) {
        ^bb0(%in: f32, %in_112: f32, %out: f32):
          %157 = arith.mulf %in, %in_112 : f32
          %158 = arith.addf %157, %out : f32
          linalg.yield %158 : f32
        } -> tensor<f32>
        %extracted_111 = tensor.extract %155[] : tensor<f32>
        %156 = arith.divf %extracted_111, %cst_2 : f32
        %inserted = tensor.insert %156 into %arg19[%arg18] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      %147 = scf.for %arg18 = %15 to %c1024 step %c1 iter_args(%arg19 = %146) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_1 into %arg19[%arg18] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      %148 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %149 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%147 : tensor<1024xf32>) outs(%148 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %155 = arith.maxnumf %in, %out : f32
        linalg.yield %155 : f32
      } -> tensor<f32>
      %extracted_105 = tensor.extract %149[] : tensor<f32>
      %150 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%147 : tensor<1024xf32>) outs(%147 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %155 = arith.subf %in, %extracted_105 : f32
        %156 = math.exp %155 : f32
        linalg.yield %156 : f32
      } -> tensor<1024xf32>
      %151 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%150 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %155 = arith.addf %in, %out : f32
        linalg.yield %155 : f32
      } -> tensor<f32>
      %extracted_106 = tensor.extract %151[] : tensor<f32>
      %152 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%150 : tensor<1024xf32>) outs(%147 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %155 = arith.divf %in, %extracted_106 : f32
        linalg.yield %155 : f32
      } -> tensor<1024xf32>
      %extracted_slice_107 = tensor.extract_slice %arg17[%144] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %153 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_107 : tensor<48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_0 : f32
      } -> tensor<48xf32>
      %inserted_slice_108 = tensor.insert_slice %153 into %arg17[%144] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %154 = scf.for %arg18 = %c0 to %15 step %c1 iter_args(%arg19 = %inserted_slice_108) -> (tensor<768xf32>) {
        %extracted_slice_109 = tensor.extract_slice %arg19[%144] [48] [1] : tensor<768xf32> to tensor<48xf32>
        %extracted_slice_110 = tensor.extract_slice %extracted_slice_15[%arg18, %144] [1, 48] [1, 1] : tensor<1024x768xf32> to tensor<48xf32>
        %extracted_111 = tensor.extract %152[%arg18] : tensor<1024xf32>
        %155 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%extracted_slice_109, %extracted_slice_110 : tensor<48xf32>, tensor<48xf32>) outs(%extracted_slice_109 : tensor<48xf32>) {
        ^bb0(%in: f32, %in_113: f32, %out: f32):
          %156 = arith.mulf %in_113, %extracted_111 : f32
          %157 = arith.addf %in, %156 : f32
          linalg.yield %157 : f32
        } -> tensor<48xf32>
        %inserted_slice_112 = tensor.insert_slice %155 into %arg19[%144] [48] [1] : tensor<48xf32> into tensor<768xf32>
        scf.yield %inserted_slice_112 : tensor<768xf32>
      }
      scf.yield %154 : tensor<768xf32>
    }
    %17 = bufferization.materialize_in_destination %16 in %14#0 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_16 = tensor.extract_slice %arg9[0, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %18 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_16, %17 : tensor<768x768xf32>, tensor<768xf32>) outs(%17 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %in_105 : f32
      %145 = arith.addf %out, %144 : f32
      linalg.yield %145 : f32
    } -> tensor<768xf32>
    %19 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%extracted_slice, %18 : tensor<768xf32>, tensor<768xf32>) outs(%18 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.addf %in, %in_105 : f32
      linalg.yield %144 : f32
    } -> tensor<768xf32>
    %extracted_slice_17 = tensor.extract_slice %arg13[0, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %20 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%19 : tensor<768xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.mulf %in, %in : f32
      %145 = arith.addf %144, %out : f32
      linalg.yield %145 : f32
    } -> tensor<f32>
    %extracted_18 = tensor.extract %20[] : tensor<f32>
    %21 = arith.divf %extracted_18, %cst_3 : f32
    %22 = arith.addf %21, %cst_4 : f32
    %23 = math.rsqrt %22 : f32
    %24 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%19, %extracted_slice_17 : tensor<768xf32>, tensor<768xf32>) outs(%6 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %23 : f32
      %145 = arith.mulf %144, %in_105 : f32
      linalg.yield %145 : f32
    } -> tensor<768xf32>
    %extracted_slice_19 = tensor.extract_slice %arg10[0, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %extracted_slice_20 = tensor.extract_slice %arg12[0, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %25 = tensor.empty() : tensor<2048xf32>
    %26 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%25 : tensor<2048xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_0 : f32
    } -> tensor<2048xf32>
    %27 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_19, %24 : tensor<2048x768xf32>, tensor<768xf32>) outs(%26 : tensor<2048xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %in_105 : f32
      %145 = arith.addf %out, %144 : f32
      linalg.yield %145 : f32
    } -> tensor<2048xf32>
    %28 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_20, %24 : tensor<2048x768xf32>, tensor<768xf32>) outs(%26 : tensor<2048xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %in_105 : f32
      %145 = arith.addf %out, %144 : f32
      linalg.yield %145 : f32
    } -> tensor<2048xf32>
    %mapped = linalg.map ins(%27, %28 : tensor<2048xf32>, tensor<2048xf32>) outs(%27 : tensor<2048xf32>)
      (%in: f32, %in_105: f32) {
        %144 = arith.negf %in : f32
        %145 = math.exp %144 : f32
        %146 = arith.addf %145, %cst_5 : f32
        %147 = arith.divf %cst_5, %146 : f32
        %148 = arith.mulf %in, %147 : f32
        %149 = arith.mulf %148, %in_105 : f32
        linalg.yield %149 : f32
      }
    %extracted_slice_21 = tensor.extract_slice %arg11[0, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
    %29 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_21, %mapped : tensor<768x2048xf32>, tensor<2048xf32>) outs(%24 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %in_105 : f32
      %145 = arith.addf %out, %144 : f32
      linalg.yield %145 : f32
    } -> tensor<768xf32>
    %extracted_slice_22 = tensor.extract_slice %arg5[1, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %30 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%29 : tensor<768xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.mulf %in, %in : f32
      %145 = arith.addf %144, %out : f32
      linalg.yield %145 : f32
    } -> tensor<f32>
    %extracted_23 = tensor.extract %30[] : tensor<f32>
    %31 = arith.divf %extracted_23, %cst_3 : f32
    %32 = arith.addf %31, %cst_4 : f32
    %33 = math.rsqrt %32 : f32
    %34 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%29, %extracted_slice_22 : tensor<768xf32>, tensor<768xf32>) outs(%6 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %33 : f32
      %145 = arith.mulf %144, %in_105 : f32
      linalg.yield %145 : f32
    } -> tensor<768xf32>
    %extracted_slice_24 = tensor.extract_slice %arg6[1, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_25 = tensor.extract_slice %arg7[1, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_26 = tensor.extract_slice %arg8[1, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %35 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_24, %34 : tensor<768x768xf32>, tensor<768xf32>) outs(%8 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %in_105 : f32
      %145 = arith.addf %out, %144 : f32
      linalg.yield %145 : f32
    } -> tensor<768xf32>
    %36 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_25, %34 : tensor<768x768xf32>, tensor<768xf32>) outs(%8 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %in_105 : f32
      %145 = arith.addf %out, %144 : f32
      linalg.yield %145 : f32
    } -> tensor<768xf32>
    %37 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_26, %34 : tensor<768x768xf32>, tensor<768xf32>) outs(%8 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %in_105 : f32
      %145 = arith.addf %out, %144 : f32
      linalg.yield %145 : f32
    } -> tensor<768xf32>
    %inserted_slice_27 = tensor.insert_slice %37 into %inserted_slice[1, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %38:2 = scf.for %arg16 = %c0 to %c768 step %c2 iter_args(%arg17 = %35, %arg18 = %36) -> (tensor<768xf32>, tensor<768xf32>) {
      %144 = arith.remui %arg16, %c48 : index
      %145 = arith.index_cast %144 : index to i64
      %146 = arith.uitofp %145 : i64 to f32
      %147 = arith.divf %146, %cst_6 : f32
      %148 = math.powf %cst_7, %147 : f32
      %149 = arith.divf %cst_5, %148 : f32
      %150 = arith.mulf %13, %149 : f32
      %151 = math.cos %150 : f32
      %152 = math.sin %150 : f32
      %153 = arith.addi %arg16, %c1 : index
      %extracted_105 = tensor.extract %arg17[%arg16] : tensor<768xf32>
      %extracted_106 = tensor.extract %arg17[%153] : tensor<768xf32>
      %154 = arith.mulf %extracted_105, %151 : f32
      %155 = arith.mulf %extracted_106, %152 : f32
      %156 = arith.subf %154, %155 : f32
      %inserted = tensor.insert %156 into %arg17[%arg16] : tensor<768xf32>
      %157 = arith.mulf %extracted_105, %152 : f32
      %158 = arith.mulf %extracted_106, %151 : f32
      %159 = arith.addf %157, %158 : f32
      %inserted_107 = tensor.insert %159 into %inserted[%153] : tensor<768xf32>
      %160 = bufferization.materialize_in_destination %inserted_107 in %arg17 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      %161 = arith.cmpi ult, %arg16, %c768 : index
      %162 = scf.if %161 -> (tensor<768xf32>) {
        %extracted_108 = tensor.extract %arg18[%arg16] : tensor<768xf32>
        %extracted_109 = tensor.extract %arg18[%153] : tensor<768xf32>
        %163 = arith.mulf %extracted_108, %151 : f32
        %164 = arith.mulf %extracted_109, %152 : f32
        %165 = arith.subf %163, %164 : f32
        %inserted_110 = tensor.insert %165 into %arg18[%arg16] : tensor<768xf32>
        %166 = arith.mulf %extracted_108, %152 : f32
        %167 = arith.mulf %extracted_109, %151 : f32
        %168 = arith.addf %166, %167 : f32
        %inserted_111 = tensor.insert %168 into %inserted_110[%153] : tensor<768xf32>
        %169 = bufferization.materialize_in_destination %inserted_111 in %arg18 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
        scf.yield %169 : tensor<768xf32>
      } else {
        scf.yield %arg18 : tensor<768xf32>
      }
      scf.yield %160, %162 : tensor<768xf32>, tensor<768xf32>
    }
    %inserted_slice_28 = tensor.insert_slice %38#1 into %inserted_slice_13[1, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %extracted_slice_29 = tensor.extract_slice %inserted_slice_28[1, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %extracted_slice_30 = tensor.extract_slice %inserted_slice_27[1, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %39 = scf.for %arg16 = %c0 to %c6 step %c1 iter_args(%arg17 = %38#0) -> (tensor<768xf32>) {
      %144 = arith.muli %arg16, %c48 : index
      %145 = tensor.empty() : tensor<1024xf32>
      %146 = scf.for %arg18 = %c0 to %15 step %c1 iter_args(%arg19 = %145) -> (tensor<1024xf32>) {
        %extracted_slice_109 = tensor.extract_slice %38#0[%144] [48] [1] : tensor<768xf32> to tensor<48xf32>
        %extracted_slice_110 = tensor.extract_slice %extracted_slice_29[%arg18, %144] [1, 48] [1, 1] : tensor<1024x768xf32> to tensor<48xf32>
        %155 = linalg.generic {indexing_maps = [#map1, #map1, #map2], iterator_types = ["reduction"]} ins(%extracted_slice_109, %extracted_slice_110 : tensor<48xf32>, tensor<48xf32>) outs(%1 : tensor<f32>) {
        ^bb0(%in: f32, %in_112: f32, %out: f32):
          %157 = arith.mulf %in, %in_112 : f32
          %158 = arith.addf %157, %out : f32
          linalg.yield %158 : f32
        } -> tensor<f32>
        %extracted_111 = tensor.extract %155[] : tensor<f32>
        %156 = arith.divf %extracted_111, %cst_2 : f32
        %inserted = tensor.insert %156 into %arg19[%arg18] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      %147 = scf.for %arg18 = %15 to %c1024 step %c1 iter_args(%arg19 = %146) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_1 into %arg19[%arg18] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      %148 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %149 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%147 : tensor<1024xf32>) outs(%148 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %155 = arith.maxnumf %in, %out : f32
        linalg.yield %155 : f32
      } -> tensor<f32>
      %extracted_105 = tensor.extract %149[] : tensor<f32>
      %150 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%147 : tensor<1024xf32>) outs(%147 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %155 = arith.subf %in, %extracted_105 : f32
        %156 = math.exp %155 : f32
        linalg.yield %156 : f32
      } -> tensor<1024xf32>
      %151 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%150 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %155 = arith.addf %in, %out : f32
        linalg.yield %155 : f32
      } -> tensor<f32>
      %extracted_106 = tensor.extract %151[] : tensor<f32>
      %152 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%150 : tensor<1024xf32>) outs(%147 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %155 = arith.divf %in, %extracted_106 : f32
        linalg.yield %155 : f32
      } -> tensor<1024xf32>
      %extracted_slice_107 = tensor.extract_slice %arg17[%144] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %153 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_107 : tensor<48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_0 : f32
      } -> tensor<48xf32>
      %inserted_slice_108 = tensor.insert_slice %153 into %arg17[%144] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %154 = scf.for %arg18 = %c0 to %15 step %c1 iter_args(%arg19 = %inserted_slice_108) -> (tensor<768xf32>) {
        %extracted_slice_109 = tensor.extract_slice %arg19[%144] [48] [1] : tensor<768xf32> to tensor<48xf32>
        %extracted_slice_110 = tensor.extract_slice %extracted_slice_30[%arg18, %144] [1, 48] [1, 1] : tensor<1024x768xf32> to tensor<48xf32>
        %extracted_111 = tensor.extract %152[%arg18] : tensor<1024xf32>
        %155 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%extracted_slice_109, %extracted_slice_110 : tensor<48xf32>, tensor<48xf32>) outs(%extracted_slice_109 : tensor<48xf32>) {
        ^bb0(%in: f32, %in_113: f32, %out: f32):
          %156 = arith.mulf %in_113, %extracted_111 : f32
          %157 = arith.addf %in, %156 : f32
          linalg.yield %157 : f32
        } -> tensor<48xf32>
        %inserted_slice_112 = tensor.insert_slice %155 into %arg19[%144] [48] [1] : tensor<48xf32> into tensor<768xf32>
        scf.yield %inserted_slice_112 : tensor<768xf32>
      }
      scf.yield %154 : tensor<768xf32>
    }
    %40 = bufferization.materialize_in_destination %39 in %38#0 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_31 = tensor.extract_slice %arg9[1, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %41 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_31, %40 : tensor<768x768xf32>, tensor<768xf32>) outs(%40 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %in_105 : f32
      %145 = arith.addf %out, %144 : f32
      linalg.yield %145 : f32
    } -> tensor<768xf32>
    %42 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%29, %41 : tensor<768xf32>, tensor<768xf32>) outs(%41 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.addf %in, %in_105 : f32
      linalg.yield %144 : f32
    } -> tensor<768xf32>
    %extracted_slice_32 = tensor.extract_slice %arg13[1, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %43 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%42 : tensor<768xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.mulf %in, %in : f32
      %145 = arith.addf %144, %out : f32
      linalg.yield %145 : f32
    } -> tensor<f32>
    %extracted_33 = tensor.extract %43[] : tensor<f32>
    %44 = arith.divf %extracted_33, %cst_3 : f32
    %45 = arith.addf %44, %cst_4 : f32
    %46 = math.rsqrt %45 : f32
    %47 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%42, %extracted_slice_32 : tensor<768xf32>, tensor<768xf32>) outs(%6 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %46 : f32
      %145 = arith.mulf %144, %in_105 : f32
      linalg.yield %145 : f32
    } -> tensor<768xf32>
    %extracted_slice_34 = tensor.extract_slice %arg10[1, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %extracted_slice_35 = tensor.extract_slice %arg12[1, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %48 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_34, %47 : tensor<2048x768xf32>, tensor<768xf32>) outs(%26 : tensor<2048xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %in_105 : f32
      %145 = arith.addf %out, %144 : f32
      linalg.yield %145 : f32
    } -> tensor<2048xf32>
    %49 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_35, %47 : tensor<2048x768xf32>, tensor<768xf32>) outs(%26 : tensor<2048xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %in_105 : f32
      %145 = arith.addf %out, %144 : f32
      linalg.yield %145 : f32
    } -> tensor<2048xf32>
    %mapped_36 = linalg.map ins(%48, %49 : tensor<2048xf32>, tensor<2048xf32>) outs(%48 : tensor<2048xf32>)
      (%in: f32, %in_105: f32) {
        %144 = arith.negf %in : f32
        %145 = math.exp %144 : f32
        %146 = arith.addf %145, %cst_5 : f32
        %147 = arith.divf %cst_5, %146 : f32
        %148 = arith.mulf %in, %147 : f32
        %149 = arith.mulf %148, %in_105 : f32
        linalg.yield %149 : f32
      }
    %extracted_slice_37 = tensor.extract_slice %arg11[1, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
    %50 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_37, %mapped_36 : tensor<768x2048xf32>, tensor<2048xf32>) outs(%47 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %in_105 : f32
      %145 = arith.addf %out, %144 : f32
      linalg.yield %145 : f32
    } -> tensor<768xf32>
    %extracted_slice_38 = tensor.extract_slice %arg5[2, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %51 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%50 : tensor<768xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.mulf %in, %in : f32
      %145 = arith.addf %144, %out : f32
      linalg.yield %145 : f32
    } -> tensor<f32>
    %extracted_39 = tensor.extract %51[] : tensor<f32>
    %52 = arith.divf %extracted_39, %cst_3 : f32
    %53 = arith.addf %52, %cst_4 : f32
    %54 = math.rsqrt %53 : f32
    %55 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%50, %extracted_slice_38 : tensor<768xf32>, tensor<768xf32>) outs(%6 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %54 : f32
      %145 = arith.mulf %144, %in_105 : f32
      linalg.yield %145 : f32
    } -> tensor<768xf32>
    %extracted_slice_40 = tensor.extract_slice %arg6[2, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_41 = tensor.extract_slice %arg7[2, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_42 = tensor.extract_slice %arg8[2, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %56 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_40, %55 : tensor<768x768xf32>, tensor<768xf32>) outs(%8 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %in_105 : f32
      %145 = arith.addf %out, %144 : f32
      linalg.yield %145 : f32
    } -> tensor<768xf32>
    %57 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_41, %55 : tensor<768x768xf32>, tensor<768xf32>) outs(%8 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %in_105 : f32
      %145 = arith.addf %out, %144 : f32
      linalg.yield %145 : f32
    } -> tensor<768xf32>
    %58 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_42, %55 : tensor<768x768xf32>, tensor<768xf32>) outs(%8 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %in_105 : f32
      %145 = arith.addf %out, %144 : f32
      linalg.yield %145 : f32
    } -> tensor<768xf32>
    %inserted_slice_43 = tensor.insert_slice %58 into %inserted_slice_27[2, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %59:2 = scf.for %arg16 = %c0 to %c768 step %c2 iter_args(%arg17 = %56, %arg18 = %57) -> (tensor<768xf32>, tensor<768xf32>) {
      %144 = arith.remui %arg16, %c48 : index
      %145 = arith.index_cast %144 : index to i64
      %146 = arith.uitofp %145 : i64 to f32
      %147 = arith.divf %146, %cst_6 : f32
      %148 = math.powf %cst_7, %147 : f32
      %149 = arith.divf %cst_5, %148 : f32
      %150 = arith.mulf %13, %149 : f32
      %151 = math.cos %150 : f32
      %152 = math.sin %150 : f32
      %153 = arith.addi %arg16, %c1 : index
      %extracted_105 = tensor.extract %arg17[%arg16] : tensor<768xf32>
      %extracted_106 = tensor.extract %arg17[%153] : tensor<768xf32>
      %154 = arith.mulf %extracted_105, %151 : f32
      %155 = arith.mulf %extracted_106, %152 : f32
      %156 = arith.subf %154, %155 : f32
      %inserted = tensor.insert %156 into %arg17[%arg16] : tensor<768xf32>
      %157 = arith.mulf %extracted_105, %152 : f32
      %158 = arith.mulf %extracted_106, %151 : f32
      %159 = arith.addf %157, %158 : f32
      %inserted_107 = tensor.insert %159 into %inserted[%153] : tensor<768xf32>
      %160 = bufferization.materialize_in_destination %inserted_107 in %arg17 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      %161 = arith.cmpi ult, %arg16, %c768 : index
      %162 = scf.if %161 -> (tensor<768xf32>) {
        %extracted_108 = tensor.extract %arg18[%arg16] : tensor<768xf32>
        %extracted_109 = tensor.extract %arg18[%153] : tensor<768xf32>
        %163 = arith.mulf %extracted_108, %151 : f32
        %164 = arith.mulf %extracted_109, %152 : f32
        %165 = arith.subf %163, %164 : f32
        %inserted_110 = tensor.insert %165 into %arg18[%arg16] : tensor<768xf32>
        %166 = arith.mulf %extracted_108, %152 : f32
        %167 = arith.mulf %extracted_109, %151 : f32
        %168 = arith.addf %166, %167 : f32
        %inserted_111 = tensor.insert %168 into %inserted_110[%153] : tensor<768xf32>
        %169 = bufferization.materialize_in_destination %inserted_111 in %arg18 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
        scf.yield %169 : tensor<768xf32>
      } else {
        scf.yield %arg18 : tensor<768xf32>
      }
      scf.yield %160, %162 : tensor<768xf32>, tensor<768xf32>
    }
    %inserted_slice_44 = tensor.insert_slice %59#1 into %inserted_slice_28[2, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %extracted_slice_45 = tensor.extract_slice %inserted_slice_44[2, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %extracted_slice_46 = tensor.extract_slice %inserted_slice_43[2, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %60 = scf.for %arg16 = %c0 to %c6 step %c1 iter_args(%arg17 = %59#0) -> (tensor<768xf32>) {
      %144 = arith.muli %arg16, %c48 : index
      %145 = tensor.empty() : tensor<1024xf32>
      %146 = scf.for %arg18 = %c0 to %15 step %c1 iter_args(%arg19 = %145) -> (tensor<1024xf32>) {
        %extracted_slice_109 = tensor.extract_slice %59#0[%144] [48] [1] : tensor<768xf32> to tensor<48xf32>
        %extracted_slice_110 = tensor.extract_slice %extracted_slice_45[%arg18, %144] [1, 48] [1, 1] : tensor<1024x768xf32> to tensor<48xf32>
        %155 = linalg.generic {indexing_maps = [#map1, #map1, #map2], iterator_types = ["reduction"]} ins(%extracted_slice_109, %extracted_slice_110 : tensor<48xf32>, tensor<48xf32>) outs(%1 : tensor<f32>) {
        ^bb0(%in: f32, %in_112: f32, %out: f32):
          %157 = arith.mulf %in, %in_112 : f32
          %158 = arith.addf %157, %out : f32
          linalg.yield %158 : f32
        } -> tensor<f32>
        %extracted_111 = tensor.extract %155[] : tensor<f32>
        %156 = arith.divf %extracted_111, %cst_2 : f32
        %inserted = tensor.insert %156 into %arg19[%arg18] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      %147 = scf.for %arg18 = %15 to %c1024 step %c1 iter_args(%arg19 = %146) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_1 into %arg19[%arg18] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      %148 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %149 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%147 : tensor<1024xf32>) outs(%148 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %155 = arith.maxnumf %in, %out : f32
        linalg.yield %155 : f32
      } -> tensor<f32>
      %extracted_105 = tensor.extract %149[] : tensor<f32>
      %150 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%147 : tensor<1024xf32>) outs(%147 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %155 = arith.subf %in, %extracted_105 : f32
        %156 = math.exp %155 : f32
        linalg.yield %156 : f32
      } -> tensor<1024xf32>
      %151 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%150 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %155 = arith.addf %in, %out : f32
        linalg.yield %155 : f32
      } -> tensor<f32>
      %extracted_106 = tensor.extract %151[] : tensor<f32>
      %152 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%150 : tensor<1024xf32>) outs(%147 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %155 = arith.divf %in, %extracted_106 : f32
        linalg.yield %155 : f32
      } -> tensor<1024xf32>
      %extracted_slice_107 = tensor.extract_slice %arg17[%144] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %153 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_107 : tensor<48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_0 : f32
      } -> tensor<48xf32>
      %inserted_slice_108 = tensor.insert_slice %153 into %arg17[%144] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %154 = scf.for %arg18 = %c0 to %15 step %c1 iter_args(%arg19 = %inserted_slice_108) -> (tensor<768xf32>) {
        %extracted_slice_109 = tensor.extract_slice %arg19[%144] [48] [1] : tensor<768xf32> to tensor<48xf32>
        %extracted_slice_110 = tensor.extract_slice %extracted_slice_46[%arg18, %144] [1, 48] [1, 1] : tensor<1024x768xf32> to tensor<48xf32>
        %extracted_111 = tensor.extract %152[%arg18] : tensor<1024xf32>
        %155 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%extracted_slice_109, %extracted_slice_110 : tensor<48xf32>, tensor<48xf32>) outs(%extracted_slice_109 : tensor<48xf32>) {
        ^bb0(%in: f32, %in_113: f32, %out: f32):
          %156 = arith.mulf %in_113, %extracted_111 : f32
          %157 = arith.addf %in, %156 : f32
          linalg.yield %157 : f32
        } -> tensor<48xf32>
        %inserted_slice_112 = tensor.insert_slice %155 into %arg19[%144] [48] [1] : tensor<48xf32> into tensor<768xf32>
        scf.yield %inserted_slice_112 : tensor<768xf32>
      }
      scf.yield %154 : tensor<768xf32>
    }
    %61 = bufferization.materialize_in_destination %60 in %59#0 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_47 = tensor.extract_slice %arg9[2, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %62 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_47, %61 : tensor<768x768xf32>, tensor<768xf32>) outs(%61 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %in_105 : f32
      %145 = arith.addf %out, %144 : f32
      linalg.yield %145 : f32
    } -> tensor<768xf32>
    %63 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%50, %62 : tensor<768xf32>, tensor<768xf32>) outs(%62 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.addf %in, %in_105 : f32
      linalg.yield %144 : f32
    } -> tensor<768xf32>
    %extracted_slice_48 = tensor.extract_slice %arg13[2, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %64 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%63 : tensor<768xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.mulf %in, %in : f32
      %145 = arith.addf %144, %out : f32
      linalg.yield %145 : f32
    } -> tensor<f32>
    %extracted_49 = tensor.extract %64[] : tensor<f32>
    %65 = arith.divf %extracted_49, %cst_3 : f32
    %66 = arith.addf %65, %cst_4 : f32
    %67 = math.rsqrt %66 : f32
    %68 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%63, %extracted_slice_48 : tensor<768xf32>, tensor<768xf32>) outs(%6 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %67 : f32
      %145 = arith.mulf %144, %in_105 : f32
      linalg.yield %145 : f32
    } -> tensor<768xf32>
    %extracted_slice_50 = tensor.extract_slice %arg10[2, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %extracted_slice_51 = tensor.extract_slice %arg12[2, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %69 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_50, %68 : tensor<2048x768xf32>, tensor<768xf32>) outs(%26 : tensor<2048xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %in_105 : f32
      %145 = arith.addf %out, %144 : f32
      linalg.yield %145 : f32
    } -> tensor<2048xf32>
    %70 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_51, %68 : tensor<2048x768xf32>, tensor<768xf32>) outs(%26 : tensor<2048xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %in_105 : f32
      %145 = arith.addf %out, %144 : f32
      linalg.yield %145 : f32
    } -> tensor<2048xf32>
    %mapped_52 = linalg.map ins(%69, %70 : tensor<2048xf32>, tensor<2048xf32>) outs(%69 : tensor<2048xf32>)
      (%in: f32, %in_105: f32) {
        %144 = arith.negf %in : f32
        %145 = math.exp %144 : f32
        %146 = arith.addf %145, %cst_5 : f32
        %147 = arith.divf %cst_5, %146 : f32
        %148 = arith.mulf %in, %147 : f32
        %149 = arith.mulf %148, %in_105 : f32
        linalg.yield %149 : f32
      }
    %extracted_slice_53 = tensor.extract_slice %arg11[2, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
    %71 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_53, %mapped_52 : tensor<768x2048xf32>, tensor<2048xf32>) outs(%68 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %in_105 : f32
      %145 = arith.addf %out, %144 : f32
      linalg.yield %145 : f32
    } -> tensor<768xf32>
    %extracted_slice_54 = tensor.extract_slice %arg5[3, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %72 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%71 : tensor<768xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.mulf %in, %in : f32
      %145 = arith.addf %144, %out : f32
      linalg.yield %145 : f32
    } -> tensor<f32>
    %extracted_55 = tensor.extract %72[] : tensor<f32>
    %73 = arith.divf %extracted_55, %cst_3 : f32
    %74 = arith.addf %73, %cst_4 : f32
    %75 = math.rsqrt %74 : f32
    %76 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%71, %extracted_slice_54 : tensor<768xf32>, tensor<768xf32>) outs(%6 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %75 : f32
      %145 = arith.mulf %144, %in_105 : f32
      linalg.yield %145 : f32
    } -> tensor<768xf32>
    %extracted_slice_56 = tensor.extract_slice %arg6[3, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_57 = tensor.extract_slice %arg7[3, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_58 = tensor.extract_slice %arg8[3, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %77 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_56, %76 : tensor<768x768xf32>, tensor<768xf32>) outs(%8 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %in_105 : f32
      %145 = arith.addf %out, %144 : f32
      linalg.yield %145 : f32
    } -> tensor<768xf32>
    %78 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_57, %76 : tensor<768x768xf32>, tensor<768xf32>) outs(%8 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %in_105 : f32
      %145 = arith.addf %out, %144 : f32
      linalg.yield %145 : f32
    } -> tensor<768xf32>
    %79 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_58, %76 : tensor<768x768xf32>, tensor<768xf32>) outs(%8 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %in_105 : f32
      %145 = arith.addf %out, %144 : f32
      linalg.yield %145 : f32
    } -> tensor<768xf32>
    %inserted_slice_59 = tensor.insert_slice %79 into %inserted_slice_43[3, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %80:2 = scf.for %arg16 = %c0 to %c768 step %c2 iter_args(%arg17 = %77, %arg18 = %78) -> (tensor<768xf32>, tensor<768xf32>) {
      %144 = arith.remui %arg16, %c48 : index
      %145 = arith.index_cast %144 : index to i64
      %146 = arith.uitofp %145 : i64 to f32
      %147 = arith.divf %146, %cst_6 : f32
      %148 = math.powf %cst_7, %147 : f32
      %149 = arith.divf %cst_5, %148 : f32
      %150 = arith.mulf %13, %149 : f32
      %151 = math.cos %150 : f32
      %152 = math.sin %150 : f32
      %153 = arith.addi %arg16, %c1 : index
      %extracted_105 = tensor.extract %arg17[%arg16] : tensor<768xf32>
      %extracted_106 = tensor.extract %arg17[%153] : tensor<768xf32>
      %154 = arith.mulf %extracted_105, %151 : f32
      %155 = arith.mulf %extracted_106, %152 : f32
      %156 = arith.subf %154, %155 : f32
      %inserted = tensor.insert %156 into %arg17[%arg16] : tensor<768xf32>
      %157 = arith.mulf %extracted_105, %152 : f32
      %158 = arith.mulf %extracted_106, %151 : f32
      %159 = arith.addf %157, %158 : f32
      %inserted_107 = tensor.insert %159 into %inserted[%153] : tensor<768xf32>
      %160 = bufferization.materialize_in_destination %inserted_107 in %arg17 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      %161 = arith.cmpi ult, %arg16, %c768 : index
      %162 = scf.if %161 -> (tensor<768xf32>) {
        %extracted_108 = tensor.extract %arg18[%arg16] : tensor<768xf32>
        %extracted_109 = tensor.extract %arg18[%153] : tensor<768xf32>
        %163 = arith.mulf %extracted_108, %151 : f32
        %164 = arith.mulf %extracted_109, %152 : f32
        %165 = arith.subf %163, %164 : f32
        %inserted_110 = tensor.insert %165 into %arg18[%arg16] : tensor<768xf32>
        %166 = arith.mulf %extracted_108, %152 : f32
        %167 = arith.mulf %extracted_109, %151 : f32
        %168 = arith.addf %166, %167 : f32
        %inserted_111 = tensor.insert %168 into %inserted_110[%153] : tensor<768xf32>
        %169 = bufferization.materialize_in_destination %inserted_111 in %arg18 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
        scf.yield %169 : tensor<768xf32>
      } else {
        scf.yield %arg18 : tensor<768xf32>
      }
      scf.yield %160, %162 : tensor<768xf32>, tensor<768xf32>
    }
    %inserted_slice_60 = tensor.insert_slice %80#1 into %inserted_slice_44[3, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %extracted_slice_61 = tensor.extract_slice %inserted_slice_60[3, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %extracted_slice_62 = tensor.extract_slice %inserted_slice_59[3, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %81 = scf.for %arg16 = %c0 to %c6 step %c1 iter_args(%arg17 = %80#0) -> (tensor<768xf32>) {
      %144 = arith.muli %arg16, %c48 : index
      %145 = tensor.empty() : tensor<1024xf32>
      %146 = scf.for %arg18 = %c0 to %15 step %c1 iter_args(%arg19 = %145) -> (tensor<1024xf32>) {
        %extracted_slice_109 = tensor.extract_slice %80#0[%144] [48] [1] : tensor<768xf32> to tensor<48xf32>
        %extracted_slice_110 = tensor.extract_slice %extracted_slice_61[%arg18, %144] [1, 48] [1, 1] : tensor<1024x768xf32> to tensor<48xf32>
        %155 = linalg.generic {indexing_maps = [#map1, #map1, #map2], iterator_types = ["reduction"]} ins(%extracted_slice_109, %extracted_slice_110 : tensor<48xf32>, tensor<48xf32>) outs(%1 : tensor<f32>) {
        ^bb0(%in: f32, %in_112: f32, %out: f32):
          %157 = arith.mulf %in, %in_112 : f32
          %158 = arith.addf %157, %out : f32
          linalg.yield %158 : f32
        } -> tensor<f32>
        %extracted_111 = tensor.extract %155[] : tensor<f32>
        %156 = arith.divf %extracted_111, %cst_2 : f32
        %inserted = tensor.insert %156 into %arg19[%arg18] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      %147 = scf.for %arg18 = %15 to %c1024 step %c1 iter_args(%arg19 = %146) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_1 into %arg19[%arg18] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      %148 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %149 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%147 : tensor<1024xf32>) outs(%148 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %155 = arith.maxnumf %in, %out : f32
        linalg.yield %155 : f32
      } -> tensor<f32>
      %extracted_105 = tensor.extract %149[] : tensor<f32>
      %150 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%147 : tensor<1024xf32>) outs(%147 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %155 = arith.subf %in, %extracted_105 : f32
        %156 = math.exp %155 : f32
        linalg.yield %156 : f32
      } -> tensor<1024xf32>
      %151 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%150 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %155 = arith.addf %in, %out : f32
        linalg.yield %155 : f32
      } -> tensor<f32>
      %extracted_106 = tensor.extract %151[] : tensor<f32>
      %152 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%150 : tensor<1024xf32>) outs(%147 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %155 = arith.divf %in, %extracted_106 : f32
        linalg.yield %155 : f32
      } -> tensor<1024xf32>
      %extracted_slice_107 = tensor.extract_slice %arg17[%144] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %153 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_107 : tensor<48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_0 : f32
      } -> tensor<48xf32>
      %inserted_slice_108 = tensor.insert_slice %153 into %arg17[%144] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %154 = scf.for %arg18 = %c0 to %15 step %c1 iter_args(%arg19 = %inserted_slice_108) -> (tensor<768xf32>) {
        %extracted_slice_109 = tensor.extract_slice %arg19[%144] [48] [1] : tensor<768xf32> to tensor<48xf32>
        %extracted_slice_110 = tensor.extract_slice %extracted_slice_62[%arg18, %144] [1, 48] [1, 1] : tensor<1024x768xf32> to tensor<48xf32>
        %extracted_111 = tensor.extract %152[%arg18] : tensor<1024xf32>
        %155 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%extracted_slice_109, %extracted_slice_110 : tensor<48xf32>, tensor<48xf32>) outs(%extracted_slice_109 : tensor<48xf32>) {
        ^bb0(%in: f32, %in_113: f32, %out: f32):
          %156 = arith.mulf %in_113, %extracted_111 : f32
          %157 = arith.addf %in, %156 : f32
          linalg.yield %157 : f32
        } -> tensor<48xf32>
        %inserted_slice_112 = tensor.insert_slice %155 into %arg19[%144] [48] [1] : tensor<48xf32> into tensor<768xf32>
        scf.yield %inserted_slice_112 : tensor<768xf32>
      }
      scf.yield %154 : tensor<768xf32>
    }
    %82 = bufferization.materialize_in_destination %81 in %80#0 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_63 = tensor.extract_slice %arg9[3, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %83 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_63, %82 : tensor<768x768xf32>, tensor<768xf32>) outs(%82 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %in_105 : f32
      %145 = arith.addf %out, %144 : f32
      linalg.yield %145 : f32
    } -> tensor<768xf32>
    %84 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%71, %83 : tensor<768xf32>, tensor<768xf32>) outs(%83 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.addf %in, %in_105 : f32
      linalg.yield %144 : f32
    } -> tensor<768xf32>
    %extracted_slice_64 = tensor.extract_slice %arg13[3, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %85 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%84 : tensor<768xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.mulf %in, %in : f32
      %145 = arith.addf %144, %out : f32
      linalg.yield %145 : f32
    } -> tensor<f32>
    %extracted_65 = tensor.extract %85[] : tensor<f32>
    %86 = arith.divf %extracted_65, %cst_3 : f32
    %87 = arith.addf %86, %cst_4 : f32
    %88 = math.rsqrt %87 : f32
    %89 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%84, %extracted_slice_64 : tensor<768xf32>, tensor<768xf32>) outs(%6 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %88 : f32
      %145 = arith.mulf %144, %in_105 : f32
      linalg.yield %145 : f32
    } -> tensor<768xf32>
    %extracted_slice_66 = tensor.extract_slice %arg10[3, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %extracted_slice_67 = tensor.extract_slice %arg12[3, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %90 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_66, %89 : tensor<2048x768xf32>, tensor<768xf32>) outs(%26 : tensor<2048xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %in_105 : f32
      %145 = arith.addf %out, %144 : f32
      linalg.yield %145 : f32
    } -> tensor<2048xf32>
    %91 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_67, %89 : tensor<2048x768xf32>, tensor<768xf32>) outs(%26 : tensor<2048xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %in_105 : f32
      %145 = arith.addf %out, %144 : f32
      linalg.yield %145 : f32
    } -> tensor<2048xf32>
    %mapped_68 = linalg.map ins(%90, %91 : tensor<2048xf32>, tensor<2048xf32>) outs(%90 : tensor<2048xf32>)
      (%in: f32, %in_105: f32) {
        %144 = arith.negf %in : f32
        %145 = math.exp %144 : f32
        %146 = arith.addf %145, %cst_5 : f32
        %147 = arith.divf %cst_5, %146 : f32
        %148 = arith.mulf %in, %147 : f32
        %149 = arith.mulf %148, %in_105 : f32
        linalg.yield %149 : f32
      }
    %extracted_slice_69 = tensor.extract_slice %arg11[3, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
    %92 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_69, %mapped_68 : tensor<768x2048xf32>, tensor<2048xf32>) outs(%89 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %in_105 : f32
      %145 = arith.addf %out, %144 : f32
      linalg.yield %145 : f32
    } -> tensor<768xf32>
    %extracted_slice_70 = tensor.extract_slice %arg5[4, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %93 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%92 : tensor<768xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.mulf %in, %in : f32
      %145 = arith.addf %144, %out : f32
      linalg.yield %145 : f32
    } -> tensor<f32>
    %extracted_71 = tensor.extract %93[] : tensor<f32>
    %94 = arith.divf %extracted_71, %cst_3 : f32
    %95 = arith.addf %94, %cst_4 : f32
    %96 = math.rsqrt %95 : f32
    %97 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%92, %extracted_slice_70 : tensor<768xf32>, tensor<768xf32>) outs(%6 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %96 : f32
      %145 = arith.mulf %144, %in_105 : f32
      linalg.yield %145 : f32
    } -> tensor<768xf32>
    %extracted_slice_72 = tensor.extract_slice %arg6[4, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_73 = tensor.extract_slice %arg7[4, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_74 = tensor.extract_slice %arg8[4, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %98 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_72, %97 : tensor<768x768xf32>, tensor<768xf32>) outs(%8 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %in_105 : f32
      %145 = arith.addf %out, %144 : f32
      linalg.yield %145 : f32
    } -> tensor<768xf32>
    %99 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_73, %97 : tensor<768x768xf32>, tensor<768xf32>) outs(%8 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %in_105 : f32
      %145 = arith.addf %out, %144 : f32
      linalg.yield %145 : f32
    } -> tensor<768xf32>
    %100 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_74, %97 : tensor<768x768xf32>, tensor<768xf32>) outs(%8 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %in_105 : f32
      %145 = arith.addf %out, %144 : f32
      linalg.yield %145 : f32
    } -> tensor<768xf32>
    %inserted_slice_75 = tensor.insert_slice %100 into %inserted_slice_59[4, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %101:2 = scf.for %arg16 = %c0 to %c768 step %c2 iter_args(%arg17 = %98, %arg18 = %99) -> (tensor<768xf32>, tensor<768xf32>) {
      %144 = arith.remui %arg16, %c48 : index
      %145 = arith.index_cast %144 : index to i64
      %146 = arith.uitofp %145 : i64 to f32
      %147 = arith.divf %146, %cst_6 : f32
      %148 = math.powf %cst_7, %147 : f32
      %149 = arith.divf %cst_5, %148 : f32
      %150 = arith.mulf %13, %149 : f32
      %151 = math.cos %150 : f32
      %152 = math.sin %150 : f32
      %153 = arith.addi %arg16, %c1 : index
      %extracted_105 = tensor.extract %arg17[%arg16] : tensor<768xf32>
      %extracted_106 = tensor.extract %arg17[%153] : tensor<768xf32>
      %154 = arith.mulf %extracted_105, %151 : f32
      %155 = arith.mulf %extracted_106, %152 : f32
      %156 = arith.subf %154, %155 : f32
      %inserted = tensor.insert %156 into %arg17[%arg16] : tensor<768xf32>
      %157 = arith.mulf %extracted_105, %152 : f32
      %158 = arith.mulf %extracted_106, %151 : f32
      %159 = arith.addf %157, %158 : f32
      %inserted_107 = tensor.insert %159 into %inserted[%153] : tensor<768xf32>
      %160 = bufferization.materialize_in_destination %inserted_107 in %arg17 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      %161 = arith.cmpi ult, %arg16, %c768 : index
      %162 = scf.if %161 -> (tensor<768xf32>) {
        %extracted_108 = tensor.extract %arg18[%arg16] : tensor<768xf32>
        %extracted_109 = tensor.extract %arg18[%153] : tensor<768xf32>
        %163 = arith.mulf %extracted_108, %151 : f32
        %164 = arith.mulf %extracted_109, %152 : f32
        %165 = arith.subf %163, %164 : f32
        %inserted_110 = tensor.insert %165 into %arg18[%arg16] : tensor<768xf32>
        %166 = arith.mulf %extracted_108, %152 : f32
        %167 = arith.mulf %extracted_109, %151 : f32
        %168 = arith.addf %166, %167 : f32
        %inserted_111 = tensor.insert %168 into %inserted_110[%153] : tensor<768xf32>
        %169 = bufferization.materialize_in_destination %inserted_111 in %arg18 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
        scf.yield %169 : tensor<768xf32>
      } else {
        scf.yield %arg18 : tensor<768xf32>
      }
      scf.yield %160, %162 : tensor<768xf32>, tensor<768xf32>
    }
    %inserted_slice_76 = tensor.insert_slice %101#1 into %inserted_slice_60[4, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %extracted_slice_77 = tensor.extract_slice %inserted_slice_76[4, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %extracted_slice_78 = tensor.extract_slice %inserted_slice_75[4, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %102 = scf.for %arg16 = %c0 to %c6 step %c1 iter_args(%arg17 = %101#0) -> (tensor<768xf32>) {
      %144 = arith.muli %arg16, %c48 : index
      %145 = tensor.empty() : tensor<1024xf32>
      %146 = scf.for %arg18 = %c0 to %15 step %c1 iter_args(%arg19 = %145) -> (tensor<1024xf32>) {
        %extracted_slice_109 = tensor.extract_slice %101#0[%144] [48] [1] : tensor<768xf32> to tensor<48xf32>
        %extracted_slice_110 = tensor.extract_slice %extracted_slice_77[%arg18, %144] [1, 48] [1, 1] : tensor<1024x768xf32> to tensor<48xf32>
        %155 = linalg.generic {indexing_maps = [#map1, #map1, #map2], iterator_types = ["reduction"]} ins(%extracted_slice_109, %extracted_slice_110 : tensor<48xf32>, tensor<48xf32>) outs(%1 : tensor<f32>) {
        ^bb0(%in: f32, %in_112: f32, %out: f32):
          %157 = arith.mulf %in, %in_112 : f32
          %158 = arith.addf %157, %out : f32
          linalg.yield %158 : f32
        } -> tensor<f32>
        %extracted_111 = tensor.extract %155[] : tensor<f32>
        %156 = arith.divf %extracted_111, %cst_2 : f32
        %inserted = tensor.insert %156 into %arg19[%arg18] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      %147 = scf.for %arg18 = %15 to %c1024 step %c1 iter_args(%arg19 = %146) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_1 into %arg19[%arg18] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      %148 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %149 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%147 : tensor<1024xf32>) outs(%148 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %155 = arith.maxnumf %in, %out : f32
        linalg.yield %155 : f32
      } -> tensor<f32>
      %extracted_105 = tensor.extract %149[] : tensor<f32>
      %150 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%147 : tensor<1024xf32>) outs(%147 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %155 = arith.subf %in, %extracted_105 : f32
        %156 = math.exp %155 : f32
        linalg.yield %156 : f32
      } -> tensor<1024xf32>
      %151 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%150 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %155 = arith.addf %in, %out : f32
        linalg.yield %155 : f32
      } -> tensor<f32>
      %extracted_106 = tensor.extract %151[] : tensor<f32>
      %152 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%150 : tensor<1024xf32>) outs(%147 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %155 = arith.divf %in, %extracted_106 : f32
        linalg.yield %155 : f32
      } -> tensor<1024xf32>
      %extracted_slice_107 = tensor.extract_slice %arg17[%144] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %153 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_107 : tensor<48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_0 : f32
      } -> tensor<48xf32>
      %inserted_slice_108 = tensor.insert_slice %153 into %arg17[%144] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %154 = scf.for %arg18 = %c0 to %15 step %c1 iter_args(%arg19 = %inserted_slice_108) -> (tensor<768xf32>) {
        %extracted_slice_109 = tensor.extract_slice %arg19[%144] [48] [1] : tensor<768xf32> to tensor<48xf32>
        %extracted_slice_110 = tensor.extract_slice %extracted_slice_78[%arg18, %144] [1, 48] [1, 1] : tensor<1024x768xf32> to tensor<48xf32>
        %extracted_111 = tensor.extract %152[%arg18] : tensor<1024xf32>
        %155 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%extracted_slice_109, %extracted_slice_110 : tensor<48xf32>, tensor<48xf32>) outs(%extracted_slice_109 : tensor<48xf32>) {
        ^bb0(%in: f32, %in_113: f32, %out: f32):
          %156 = arith.mulf %in_113, %extracted_111 : f32
          %157 = arith.addf %in, %156 : f32
          linalg.yield %157 : f32
        } -> tensor<48xf32>
        %inserted_slice_112 = tensor.insert_slice %155 into %arg19[%144] [48] [1] : tensor<48xf32> into tensor<768xf32>
        scf.yield %inserted_slice_112 : tensor<768xf32>
      }
      scf.yield %154 : tensor<768xf32>
    }
    %103 = bufferization.materialize_in_destination %102 in %101#0 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_79 = tensor.extract_slice %arg9[4, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %104 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_79, %103 : tensor<768x768xf32>, tensor<768xf32>) outs(%103 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %in_105 : f32
      %145 = arith.addf %out, %144 : f32
      linalg.yield %145 : f32
    } -> tensor<768xf32>
    %105 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%92, %104 : tensor<768xf32>, tensor<768xf32>) outs(%104 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.addf %in, %in_105 : f32
      linalg.yield %144 : f32
    } -> tensor<768xf32>
    %extracted_slice_80 = tensor.extract_slice %arg13[4, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %106 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%105 : tensor<768xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.mulf %in, %in : f32
      %145 = arith.addf %144, %out : f32
      linalg.yield %145 : f32
    } -> tensor<f32>
    %extracted_81 = tensor.extract %106[] : tensor<f32>
    %107 = arith.divf %extracted_81, %cst_3 : f32
    %108 = arith.addf %107, %cst_4 : f32
    %109 = math.rsqrt %108 : f32
    %110 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%105, %extracted_slice_80 : tensor<768xf32>, tensor<768xf32>) outs(%6 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %109 : f32
      %145 = arith.mulf %144, %in_105 : f32
      linalg.yield %145 : f32
    } -> tensor<768xf32>
    %extracted_slice_82 = tensor.extract_slice %arg10[4, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %extracted_slice_83 = tensor.extract_slice %arg12[4, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %111 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_82, %110 : tensor<2048x768xf32>, tensor<768xf32>) outs(%26 : tensor<2048xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %in_105 : f32
      %145 = arith.addf %out, %144 : f32
      linalg.yield %145 : f32
    } -> tensor<2048xf32>
    %112 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_83, %110 : tensor<2048x768xf32>, tensor<768xf32>) outs(%26 : tensor<2048xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %in_105 : f32
      %145 = arith.addf %out, %144 : f32
      linalg.yield %145 : f32
    } -> tensor<2048xf32>
    %mapped_84 = linalg.map ins(%111, %112 : tensor<2048xf32>, tensor<2048xf32>) outs(%111 : tensor<2048xf32>)
      (%in: f32, %in_105: f32) {
        %144 = arith.negf %in : f32
        %145 = math.exp %144 : f32
        %146 = arith.addf %145, %cst_5 : f32
        %147 = arith.divf %cst_5, %146 : f32
        %148 = arith.mulf %in, %147 : f32
        %149 = arith.mulf %148, %in_105 : f32
        linalg.yield %149 : f32
      }
    %extracted_slice_85 = tensor.extract_slice %arg11[4, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
    %113 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_85, %mapped_84 : tensor<768x2048xf32>, tensor<2048xf32>) outs(%110 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %in_105 : f32
      %145 = arith.addf %out, %144 : f32
      linalg.yield %145 : f32
    } -> tensor<768xf32>
    %extracted_slice_86 = tensor.extract_slice %arg5[5, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %114 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%113 : tensor<768xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.mulf %in, %in : f32
      %145 = arith.addf %144, %out : f32
      linalg.yield %145 : f32
    } -> tensor<f32>
    %extracted_87 = tensor.extract %114[] : tensor<f32>
    %115 = arith.divf %extracted_87, %cst_3 : f32
    %116 = arith.addf %115, %cst_4 : f32
    %117 = math.rsqrt %116 : f32
    %118 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%113, %extracted_slice_86 : tensor<768xf32>, tensor<768xf32>) outs(%6 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %117 : f32
      %145 = arith.mulf %144, %in_105 : f32
      linalg.yield %145 : f32
    } -> tensor<768xf32>
    %extracted_slice_88 = tensor.extract_slice %arg6[5, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_89 = tensor.extract_slice %arg7[5, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_90 = tensor.extract_slice %arg8[5, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %119 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_88, %118 : tensor<768x768xf32>, tensor<768xf32>) outs(%8 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %in_105 : f32
      %145 = arith.addf %out, %144 : f32
      linalg.yield %145 : f32
    } -> tensor<768xf32>
    %120 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_89, %118 : tensor<768x768xf32>, tensor<768xf32>) outs(%8 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %in_105 : f32
      %145 = arith.addf %out, %144 : f32
      linalg.yield %145 : f32
    } -> tensor<768xf32>
    %121 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_90, %118 : tensor<768x768xf32>, tensor<768xf32>) outs(%8 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %in_105 : f32
      %145 = arith.addf %out, %144 : f32
      linalg.yield %145 : f32
    } -> tensor<768xf32>
    %inserted_slice_91 = tensor.insert_slice %121 into %inserted_slice_75[5, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %122:2 = scf.for %arg16 = %c0 to %c768 step %c2 iter_args(%arg17 = %119, %arg18 = %120) -> (tensor<768xf32>, tensor<768xf32>) {
      %144 = arith.remui %arg16, %c48 : index
      %145 = arith.index_cast %144 : index to i64
      %146 = arith.uitofp %145 : i64 to f32
      %147 = arith.divf %146, %cst_6 : f32
      %148 = math.powf %cst_7, %147 : f32
      %149 = arith.divf %cst_5, %148 : f32
      %150 = arith.mulf %13, %149 : f32
      %151 = math.cos %150 : f32
      %152 = math.sin %150 : f32
      %153 = arith.addi %arg16, %c1 : index
      %extracted_105 = tensor.extract %arg17[%arg16] : tensor<768xf32>
      %extracted_106 = tensor.extract %arg17[%153] : tensor<768xf32>
      %154 = arith.mulf %extracted_105, %151 : f32
      %155 = arith.mulf %extracted_106, %152 : f32
      %156 = arith.subf %154, %155 : f32
      %inserted = tensor.insert %156 into %arg17[%arg16] : tensor<768xf32>
      %157 = arith.mulf %extracted_105, %152 : f32
      %158 = arith.mulf %extracted_106, %151 : f32
      %159 = arith.addf %157, %158 : f32
      %inserted_107 = tensor.insert %159 into %inserted[%153] : tensor<768xf32>
      %160 = bufferization.materialize_in_destination %inserted_107 in %arg17 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      %161 = arith.cmpi ult, %arg16, %c768 : index
      %162 = scf.if %161 -> (tensor<768xf32>) {
        %extracted_108 = tensor.extract %arg18[%arg16] : tensor<768xf32>
        %extracted_109 = tensor.extract %arg18[%153] : tensor<768xf32>
        %163 = arith.mulf %extracted_108, %151 : f32
        %164 = arith.mulf %extracted_109, %152 : f32
        %165 = arith.subf %163, %164 : f32
        %inserted_110 = tensor.insert %165 into %arg18[%arg16] : tensor<768xf32>
        %166 = arith.mulf %extracted_108, %152 : f32
        %167 = arith.mulf %extracted_109, %151 : f32
        %168 = arith.addf %166, %167 : f32
        %inserted_111 = tensor.insert %168 into %inserted_110[%153] : tensor<768xf32>
        %169 = bufferization.materialize_in_destination %inserted_111 in %arg18 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
        scf.yield %169 : tensor<768xf32>
      } else {
        scf.yield %arg18 : tensor<768xf32>
      }
      scf.yield %160, %162 : tensor<768xf32>, tensor<768xf32>
    }
    %inserted_slice_92 = tensor.insert_slice %122#1 into %inserted_slice_76[5, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %extracted_slice_93 = tensor.extract_slice %inserted_slice_92[5, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %extracted_slice_94 = tensor.extract_slice %inserted_slice_91[5, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %123 = scf.for %arg16 = %c0 to %c6 step %c1 iter_args(%arg17 = %122#0) -> (tensor<768xf32>) {
      %144 = arith.muli %arg16, %c48 : index
      %145 = tensor.empty() : tensor<1024xf32>
      %146 = scf.for %arg18 = %c0 to %15 step %c1 iter_args(%arg19 = %145) -> (tensor<1024xf32>) {
        %extracted_slice_109 = tensor.extract_slice %122#0[%144] [48] [1] : tensor<768xf32> to tensor<48xf32>
        %extracted_slice_110 = tensor.extract_slice %extracted_slice_93[%arg18, %144] [1, 48] [1, 1] : tensor<1024x768xf32> to tensor<48xf32>
        %155 = linalg.generic {indexing_maps = [#map1, #map1, #map2], iterator_types = ["reduction"]} ins(%extracted_slice_109, %extracted_slice_110 : tensor<48xf32>, tensor<48xf32>) outs(%1 : tensor<f32>) {
        ^bb0(%in: f32, %in_112: f32, %out: f32):
          %157 = arith.mulf %in, %in_112 : f32
          %158 = arith.addf %157, %out : f32
          linalg.yield %158 : f32
        } -> tensor<f32>
        %extracted_111 = tensor.extract %155[] : tensor<f32>
        %156 = arith.divf %extracted_111, %cst_2 : f32
        %inserted = tensor.insert %156 into %arg19[%arg18] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      %147 = scf.for %arg18 = %15 to %c1024 step %c1 iter_args(%arg19 = %146) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_1 into %arg19[%arg18] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      %148 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %149 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%147 : tensor<1024xf32>) outs(%148 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %155 = arith.maxnumf %in, %out : f32
        linalg.yield %155 : f32
      } -> tensor<f32>
      %extracted_105 = tensor.extract %149[] : tensor<f32>
      %150 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%147 : tensor<1024xf32>) outs(%147 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %155 = arith.subf %in, %extracted_105 : f32
        %156 = math.exp %155 : f32
        linalg.yield %156 : f32
      } -> tensor<1024xf32>
      %151 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%150 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %155 = arith.addf %in, %out : f32
        linalg.yield %155 : f32
      } -> tensor<f32>
      %extracted_106 = tensor.extract %151[] : tensor<f32>
      %152 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%150 : tensor<1024xf32>) outs(%147 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %155 = arith.divf %in, %extracted_106 : f32
        linalg.yield %155 : f32
      } -> tensor<1024xf32>
      %extracted_slice_107 = tensor.extract_slice %arg17[%144] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %153 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_107 : tensor<48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_0 : f32
      } -> tensor<48xf32>
      %inserted_slice_108 = tensor.insert_slice %153 into %arg17[%144] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %154 = scf.for %arg18 = %c0 to %15 step %c1 iter_args(%arg19 = %inserted_slice_108) -> (tensor<768xf32>) {
        %extracted_slice_109 = tensor.extract_slice %arg19[%144] [48] [1] : tensor<768xf32> to tensor<48xf32>
        %extracted_slice_110 = tensor.extract_slice %extracted_slice_94[%arg18, %144] [1, 48] [1, 1] : tensor<1024x768xf32> to tensor<48xf32>
        %extracted_111 = tensor.extract %152[%arg18] : tensor<1024xf32>
        %155 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%extracted_slice_109, %extracted_slice_110 : tensor<48xf32>, tensor<48xf32>) outs(%extracted_slice_109 : tensor<48xf32>) {
        ^bb0(%in: f32, %in_113: f32, %out: f32):
          %156 = arith.mulf %in_113, %extracted_111 : f32
          %157 = arith.addf %in, %156 : f32
          linalg.yield %157 : f32
        } -> tensor<48xf32>
        %inserted_slice_112 = tensor.insert_slice %155 into %arg19[%144] [48] [1] : tensor<48xf32> into tensor<768xf32>
        scf.yield %inserted_slice_112 : tensor<768xf32>
      }
      scf.yield %154 : tensor<768xf32>
    }
    %124 = bufferization.materialize_in_destination %123 in %122#0 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_95 = tensor.extract_slice %arg9[5, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %125 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_95, %124 : tensor<768x768xf32>, tensor<768xf32>) outs(%124 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %in_105 : f32
      %145 = arith.addf %out, %144 : f32
      linalg.yield %145 : f32
    } -> tensor<768xf32>
    %126 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%113, %125 : tensor<768xf32>, tensor<768xf32>) outs(%125 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.addf %in, %in_105 : f32
      linalg.yield %144 : f32
    } -> tensor<768xf32>
    %extracted_slice_96 = tensor.extract_slice %arg13[5, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %127 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%126 : tensor<768xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.mulf %in, %in : f32
      %145 = arith.addf %144, %out : f32
      linalg.yield %145 : f32
    } -> tensor<f32>
    %extracted_97 = tensor.extract %127[] : tensor<f32>
    %128 = arith.divf %extracted_97, %cst_3 : f32
    %129 = arith.addf %128, %cst_4 : f32
    %130 = math.rsqrt %129 : f32
    %131 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%126, %extracted_slice_96 : tensor<768xf32>, tensor<768xf32>) outs(%6 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %130 : f32
      %145 = arith.mulf %144, %in_105 : f32
      linalg.yield %145 : f32
    } -> tensor<768xf32>
    %extracted_slice_98 = tensor.extract_slice %arg10[5, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %extracted_slice_99 = tensor.extract_slice %arg12[5, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %132 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_98, %131 : tensor<2048x768xf32>, tensor<768xf32>) outs(%26 : tensor<2048xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %in_105 : f32
      %145 = arith.addf %out, %144 : f32
      linalg.yield %145 : f32
    } -> tensor<2048xf32>
    %133 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_99, %131 : tensor<2048x768xf32>, tensor<768xf32>) outs(%26 : tensor<2048xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %in_105 : f32
      %145 = arith.addf %out, %144 : f32
      linalg.yield %145 : f32
    } -> tensor<2048xf32>
    %mapped_100 = linalg.map ins(%132, %133 : tensor<2048xf32>, tensor<2048xf32>) outs(%132 : tensor<2048xf32>)
      (%in: f32, %in_105: f32) {
        %144 = arith.negf %in : f32
        %145 = math.exp %144 : f32
        %146 = arith.addf %145, %cst_5 : f32
        %147 = arith.divf %cst_5, %146 : f32
        %148 = arith.mulf %in, %147 : f32
        %149 = arith.mulf %148, %in_105 : f32
        linalg.yield %149 : f32
      }
    %extracted_slice_101 = tensor.extract_slice %arg11[5, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
    %134 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_101, %mapped_100 : tensor<768x2048xf32>, tensor<2048xf32>) outs(%131 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_105: f32, %out: f32):
      %144 = arith.mulf %in, %in_105 : f32
      %145 = arith.addf %out, %144 : f32
      linalg.yield %145 : f32
    } -> tensor<768xf32>
    %135 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%134 : tensor<768xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.mulf %in, %in : f32
      %145 = arith.addf %144, %out : f32
      linalg.yield %145 : f32
    } -> tensor<f32>
    %extracted_102 = tensor.extract %135[] : tensor<f32>
    %136 = arith.divf %extracted_102, %cst_3 : f32
    %137 = arith.addf %136, %cst_4 : f32
    %138 = math.rsqrt %137 : f32
    %139 = tensor.empty() : tensor<34048x768xf32>
    %140 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%139 : tensor<34048x768xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_0 : f32
    } -> tensor<34048x768xf32>
    %inserted_slice_103 = tensor.insert_slice %arg15 into %140[0, 0] [32000, 768] [1, 1] : tensor<32000x768xf32> into tensor<34048x768xf32>
    %141 = tensor.empty() : tensor<34048xf32>
    %142 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%141 : tensor<34048xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_0 : f32
    } -> tensor<34048xf32>
    %143 = linalg.generic {indexing_maps = [#map3, #map4, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%inserted_slice_103, %134, %arg14 : tensor<34048x768xf32>, tensor<768xf32>, tensor<768xf32>) outs(%142 : tensor<34048xf32>) {
    ^bb0(%in: f32, %in_105: f32, %in_106: f32, %out: f32):
      %144 = arith.mulf %in_105, %138 : f32
      %145 = arith.mulf %144, %in_106 : f32
      %146 = arith.mulf %in, %145 : f32
      %147 = arith.addf %out, %146 : f32
      linalg.yield %147 : f32
    } -> tensor<34048xf32>
    %extracted_slice_104 = tensor.extract_slice %143[0] [32000] [1] : tensor<34048xf32> to tensor<32000xf32>
    return %extracted_slice_104 : tensor<32000xf32>
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
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    %c48 = arith.constant 48 : index
    %c1024 = arith.constant 1024 : index
    %cst_1 = arith.constant 6.92820311 : f32
    %cst_2 = arith.constant 0xFF800000 : f32
    %0 = arith.addi %arg3, %c1 : index
    %1 = tensor.empty() : tensor<768xf32>
    %2 = scf.for %arg4 = %c0 to %c6 step %c1 iter_args(%arg5 = %1) -> (tensor<768xf32>) {
      %3 = arith.muli %arg4, %c48 : index
      %4 = tensor.empty() : tensor<1024xf32>
      %5 = scf.for %arg6 = %c0 to %0 step %c1 iter_args(%arg7 = %4) -> (tensor<1024xf32>) {
        %extracted_slice_4 = tensor.extract_slice %arg0[%3] [48] [1] : tensor<768xf32> to tensor<48xf32>
        %extracted_slice_5 = tensor.extract_slice %arg1[%arg6, %3] [1, 48] [1, 1] : tensor<1024x768xf32> to tensor<48xf32>
        %16 = tensor.empty() : tensor<f32>
        %17 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%16 : tensor<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_0 : f32
        } -> tensor<f32>
        %18 = linalg.generic {indexing_maps = [#map1, #map1, #map2], iterator_types = ["reduction"]} ins(%extracted_slice_4, %extracted_slice_5 : tensor<48xf32>, tensor<48xf32>) outs(%17 : tensor<f32>) {
        ^bb0(%in: f32, %in_7: f32, %out: f32):
          %20 = arith.mulf %in, %in_7 : f32
          %21 = arith.addf %20, %out : f32
          linalg.yield %21 : f32
        } -> tensor<f32>
        %extracted_6 = tensor.extract %18[] : tensor<f32>
        %19 = arith.divf %extracted_6, %cst_1 : f32
        %inserted = tensor.insert %19 into %arg7[%arg6] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      %6 = scf.for %arg6 = %0 to %c1024 step %c1 iter_args(%arg7 = %5) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_2 into %arg7[%arg6] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      %7 = tensor.empty() : tensor<f32>
      %8 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%7 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %9 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%6 : tensor<1024xf32>) outs(%8 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %16 = arith.maxnumf %in, %out : f32
        linalg.yield %16 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %9[] : tensor<f32>
      %10 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%6 : tensor<1024xf32>) outs(%6 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %16 = arith.subf %in, %extracted : f32
        %17 = math.exp %16 : f32
        linalg.yield %17 : f32
      } -> tensor<1024xf32>
      %11 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%7 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_0 : f32
      } -> tensor<f32>
      %12 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%10 : tensor<1024xf32>) outs(%11 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %16 = arith.addf %in, %out : f32
        linalg.yield %16 : f32
      } -> tensor<f32>
      %extracted_3 = tensor.extract %12[] : tensor<f32>
      %13 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%10 : tensor<1024xf32>) outs(%6 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %16 = arith.divf %in, %extracted_3 : f32
        linalg.yield %16 : f32
      } -> tensor<1024xf32>
      %extracted_slice = tensor.extract_slice %arg5[%3] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %14 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice : tensor<48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_0 : f32
      } -> tensor<48xf32>
      %inserted_slice = tensor.insert_slice %14 into %arg5[%3] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %15 = scf.for %arg6 = %c0 to %0 step %c1 iter_args(%arg7 = %inserted_slice) -> (tensor<768xf32>) {
        %extracted_slice_4 = tensor.extract_slice %arg7[%3] [48] [1] : tensor<768xf32> to tensor<48xf32>
        %extracted_slice_5 = tensor.extract_slice %arg2[%arg6, %3] [1, 48] [1, 1] : tensor<1024x768xf32> to tensor<48xf32>
        %extracted_6 = tensor.extract %13[%arg6] : tensor<1024xf32>
        %16 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%extracted_slice_4, %extracted_slice_5 : tensor<48xf32>, tensor<48xf32>) outs(%extracted_slice_4 : tensor<48xf32>) {
        ^bb0(%in: f32, %in_8: f32, %out: f32):
          %17 = arith.mulf %in_8, %extracted_6 : f32
          %18 = arith.addf %in, %17 : f32
          linalg.yield %18 : f32
        } -> tensor<48xf32>
        %inserted_slice_7 = tensor.insert_slice %16 into %arg7[%3] [48] [1] : tensor<48xf32> into tensor<768xf32>
        scf.yield %inserted_slice_7 : tensor<768xf32>
      }
      scf.yield %15 : tensor<768xf32>
    }
    return %2 : tensor<768xf32>
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
    %7 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<768xf32>, tensor<768xf32>) outs(%6 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %8 = arith.mulf %in, %5 : f32
      %9 = arith.mulf %8, %in_2 : f32
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
    %3 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%arg0 : tensor<1024xf32>) outs(%arg0 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %7 = arith.subf %in, %extracted : f32
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
    %6 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%3 : tensor<1024xf32>) outs(%arg0 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %7 = arith.divf %in, %extracted_1 : f32
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
      transform.yield 
    }
  }
}

