#upmem = #upmem.platform<type = v1A, dpus = 2560, tasklets = 24>
module {
  func.func @forward(%arg0: index, %arg1: index, %arg2: tensor<6x1024x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, bufferization.writable = true}, %arg3: tensor<6x1024x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, bufferization.writable = true}, %arg4: tensor<32000x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, cinm.static}, %arg5: tensor<6x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, cinm.static}, %arg6: tensor<6x768x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, cinm.static}, %arg7: tensor<6x768x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, cinm.static}, %arg8: tensor<6x768x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, cinm.static}, %arg9: tensor<6x768x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, cinm.static}, %arg10: tensor<6x2048x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, cinm.static}, %arg11: tensor<6x768x2048xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, cinm.static}, %arg12: tensor<6x2048x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, cinm.static}, %arg13: tensor<6x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, cinm.static}, %arg14: tensor<768xf32> {bufferization.buffer_layout = affine_map<(d0) -> (d0)>, cinm.static}, %arg15: tensor<32000x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, cinm.static}) -> tensor<32000xf32> attributes {cinm.available_platforms = [#upmem]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c6 = arith.constant 6 : index
    %c48 = arith.constant 48 : index
    %c768 = arith.constant 768 : index
    %c2048 = arith.constant 2048 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 4.800000e+01 : f32
    %cst_2 = arith.constant 1.000000e+04 : f32
    %extracted_slice = tensor.extract_slice %arg4[%arg0, 0] [1, 768] [1, 1] : tensor<32000x768xf32> to tensor<768xf32>
    %c6_3 = arith.constant 6 : index
    %extracted_slice_4 = tensor.extract_slice %arg5[%c0, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %0 = call @rmsnorm(%extracted_slice, %extracted_slice_4) : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_5 = tensor.extract_slice %arg6[%c0, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_6 = tensor.extract_slice %arg7[%c0, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_7 = tensor.extract_slice %arg8[%c0, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_8 = tensor.extract_slice %arg2[%c0, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %extracted_slice_9 = tensor.extract_slice %arg3[%c0, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %1 = cinm.op.gemv %extracted_slice_5, %0 : tensor<768x768xf32>, tensor<768xf32> -> tensor<768xf32>
    %2 = cinm.op.gemv %extracted_slice_6, %0 : tensor<768x768xf32>, tensor<768xf32> -> tensor<768xf32>
    %3 = cinm.op.gemv %extracted_slice_7, %0 : tensor<768x768xf32>, tensor<768xf32> -> tensor<768xf32>
    %inserted_slice = tensor.insert_slice %2 into %arg2[%c0, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %inserted_slice_10 = tensor.insert_slice %3 into %arg3[%c0, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %4 = arith.index_cast %arg1 : index to i64
    %5 = arith.uitofp %4 : i64 to f32
    %6:2 = scf.for %arg16 = %c0 to %c768 step %c2 iter_args(%arg17 = %1, %arg18 = %2) -> (tensor<768xf32>, tensor<768xf32>) {
      %102 = arith.remui %arg16, %c48 : index
      %103 = arith.index_cast %102 : index to i64
      %104 = arith.uitofp %103 : i64 to f32
      %105 = arith.divf %104, %cst_1 : f32
      %106 = math.powf %cst_2, %105 : f32
      %107 = arith.divf %cst_0, %106 : f32
      %108 = arith.mulf %5, %107 : f32
      %109 = math.cos %108 : f32
      %110 = math.sin %108 : f32
      %111 = func.call @rot(%arg17, %arg16, %109, %110) : (tensor<768xf32>, index, f32, f32) -> tensor<768xf32>
      %112 = arith.cmpi ult, %arg16, %c768 : index
      %113 = scf.if %112 -> (tensor<768xf32>) {
        %114 = func.call @rot(%arg18, %arg16, %109, %110) : (tensor<768xf32>, index, f32, f32) -> tensor<768xf32>
        scf.yield %114 : tensor<768xf32>
      } else {
        scf.yield %arg18 : tensor<768xf32>
      }
      scf.yield %111, %113 : tensor<768xf32>, tensor<768xf32>
    }
    %inserted_slice_11 = tensor.insert_slice %6#1 into %inserted_slice[%c0, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %extracted_slice_12 = tensor.extract_slice %inserted_slice_11[%c0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %extracted_slice_13 = tensor.extract_slice %inserted_slice_10[%c0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %7 = call @mha(%6#0, %extracted_slice_12, %extracted_slice_13, %arg1) : (tensor<768xf32>, tensor<1024x768xf32>, tensor<1024x768xf32>, index) -> tensor<768xf32>
    %8 = bufferization.materialize_in_destination %7 in %0 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_14 = tensor.extract_slice %arg9[%c0, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %9 = cinm.op.gemv %extracted_slice_14, %8 plus %extracted_slice into %extracted_slice : tensor<768x768xf32>, tensor<768xf32> plus tensor<768xf32> into tensor<768xf32> -> tensor<768xf32>
    %extracted_slice_15 = tensor.extract_slice %arg13[%c0, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %10 = call @rmsnorm(%9, %extracted_slice_15) : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %11 = bufferization.materialize_in_destination %10 in %8 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_16 = tensor.extract_slice %arg10[%c0, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %extracted_slice_17 = tensor.extract_slice %arg12[%c0, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %12 = cinm.op.gemv %extracted_slice_16, %11 : tensor<2048x768xf32>, tensor<768xf32> -> tensor<2048xf32>
    %13 = cinm.op.gemv %extracted_slice_17, %11 : tensor<2048x768xf32>, tensor<768xf32> -> tensor<2048xf32>
    %mapped = linalg.map ins(%13 : tensor<2048xf32>) outs(%12 : tensor<2048xf32>)
      (%in: f32, %init: f32) {
        %102 = arith.negf %init : f32
        %103 = math.exp %102 : f32
        %104 = arith.addf %cst_0, %103 : f32
        %105 = arith.divf %cst_0, %104 : f32
        %106 = arith.mulf %init, %105 : f32
        %107 = arith.mulf %106, %in : f32
        linalg.yield %107 : f32
      }
    %extracted_slice_18 = tensor.extract_slice %arg11[%c0, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
    %14 = cinm.op.gemv %extracted_slice_18, %mapped plus %11 into %9 : tensor<768x2048xf32>, tensor<2048xf32> plus tensor<768xf32> into tensor<768xf32> -> tensor<768xf32>
    %c1_19 = arith.constant 1 : index
    %15 = arith.muli %c1, %c1_19 : index
    %16 = arith.addi %c0, %15 : index
    %extracted_slice_20 = tensor.extract_slice %arg5[%16, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %17 = call @rmsnorm(%14, %extracted_slice_20) : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_21 = tensor.extract_slice %arg6[%16, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_22 = tensor.extract_slice %arg7[%16, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_23 = tensor.extract_slice %arg8[%16, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_24 = tensor.extract_slice %inserted_slice_11[%16, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %extracted_slice_25 = tensor.extract_slice %inserted_slice_10[%16, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %18 = cinm.op.gemv %extracted_slice_21, %17 : tensor<768x768xf32>, tensor<768xf32> -> tensor<768xf32>
    %19 = cinm.op.gemv %extracted_slice_22, %17 : tensor<768x768xf32>, tensor<768xf32> -> tensor<768xf32>
    %20 = cinm.op.gemv %extracted_slice_23, %17 : tensor<768x768xf32>, tensor<768xf32> -> tensor<768xf32>
    %inserted_slice_26 = tensor.insert_slice %19 into %inserted_slice_11[%16, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %inserted_slice_27 = tensor.insert_slice %20 into %inserted_slice_10[%16, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %21 = arith.index_cast %arg1 : index to i64
    %22 = arith.uitofp %21 : i64 to f32
    %23:2 = scf.for %arg16 = %c0 to %c768 step %c2 iter_args(%arg17 = %18, %arg18 = %19) -> (tensor<768xf32>, tensor<768xf32>) {
      %102 = arith.remui %arg16, %c48 : index
      %103 = arith.index_cast %102 : index to i64
      %104 = arith.uitofp %103 : i64 to f32
      %105 = arith.divf %104, %cst_1 : f32
      %106 = math.powf %cst_2, %105 : f32
      %107 = arith.divf %cst_0, %106 : f32
      %108 = arith.mulf %22, %107 : f32
      %109 = math.cos %108 : f32
      %110 = math.sin %108 : f32
      %111 = func.call @rot(%arg17, %arg16, %109, %110) : (tensor<768xf32>, index, f32, f32) -> tensor<768xf32>
      %112 = arith.cmpi ult, %arg16, %c768 : index
      %113 = scf.if %112 -> (tensor<768xf32>) {
        %114 = func.call @rot(%arg18, %arg16, %109, %110) : (tensor<768xf32>, index, f32, f32) -> tensor<768xf32>
        scf.yield %114 : tensor<768xf32>
      } else {
        scf.yield %arg18 : tensor<768xf32>
      }
      scf.yield %111, %113 : tensor<768xf32>, tensor<768xf32>
    }
    %inserted_slice_28 = tensor.insert_slice %23#1 into %inserted_slice_26[%16, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %extracted_slice_29 = tensor.extract_slice %inserted_slice_28[%16, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %extracted_slice_30 = tensor.extract_slice %inserted_slice_27[%16, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %24 = call @mha(%23#0, %extracted_slice_29, %extracted_slice_30, %arg1) : (tensor<768xf32>, tensor<1024x768xf32>, tensor<1024x768xf32>, index) -> tensor<768xf32>
    %25 = bufferization.materialize_in_destination %24 in %17 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_31 = tensor.extract_slice %arg9[%16, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %26 = cinm.op.gemv %extracted_slice_31, %25 plus %14 into %14 : tensor<768x768xf32>, tensor<768xf32> plus tensor<768xf32> into tensor<768xf32> -> tensor<768xf32>
    %extracted_slice_32 = tensor.extract_slice %arg13[%16, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %27 = call @rmsnorm(%26, %extracted_slice_32) : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %28 = bufferization.materialize_in_destination %27 in %25 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_33 = tensor.extract_slice %arg10[%16, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %extracted_slice_34 = tensor.extract_slice %arg12[%16, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %29 = cinm.op.gemv %extracted_slice_33, %28 : tensor<2048x768xf32>, tensor<768xf32> -> tensor<2048xf32>
    %30 = cinm.op.gemv %extracted_slice_34, %28 : tensor<2048x768xf32>, tensor<768xf32> -> tensor<2048xf32>
    %mapped_35 = linalg.map ins(%30 : tensor<2048xf32>) outs(%29 : tensor<2048xf32>)
      (%in: f32, %init: f32) {
        %102 = arith.negf %init : f32
        %103 = math.exp %102 : f32
        %104 = arith.addf %cst_0, %103 : f32
        %105 = arith.divf %cst_0, %104 : f32
        %106 = arith.mulf %init, %105 : f32
        %107 = arith.mulf %106, %in : f32
        linalg.yield %107 : f32
      }
    %extracted_slice_36 = tensor.extract_slice %arg11[%16, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
    %31 = cinm.op.gemv %extracted_slice_36, %mapped_35 plus %28 into %26 : tensor<768x2048xf32>, tensor<2048xf32> plus tensor<768xf32> into tensor<768xf32> -> tensor<768xf32>
    %c2_37 = arith.constant 2 : index
    %32 = arith.muli %c1, %c2_37 : index
    %33 = arith.addi %c0, %32 : index
    %extracted_slice_38 = tensor.extract_slice %arg5[%33, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %34 = call @rmsnorm(%31, %extracted_slice_38) : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_39 = tensor.extract_slice %arg6[%33, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_40 = tensor.extract_slice %arg7[%33, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_41 = tensor.extract_slice %arg8[%33, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_42 = tensor.extract_slice %inserted_slice_28[%33, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %extracted_slice_43 = tensor.extract_slice %inserted_slice_27[%33, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %35 = cinm.op.gemv %extracted_slice_39, %34 : tensor<768x768xf32>, tensor<768xf32> -> tensor<768xf32>
    %36 = cinm.op.gemv %extracted_slice_40, %34 : tensor<768x768xf32>, tensor<768xf32> -> tensor<768xf32>
    %37 = cinm.op.gemv %extracted_slice_41, %34 : tensor<768x768xf32>, tensor<768xf32> -> tensor<768xf32>
    %inserted_slice_44 = tensor.insert_slice %36 into %inserted_slice_28[%33, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %inserted_slice_45 = tensor.insert_slice %37 into %inserted_slice_27[%33, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %38 = arith.index_cast %arg1 : index to i64
    %39 = arith.uitofp %38 : i64 to f32
    %40:2 = scf.for %arg16 = %c0 to %c768 step %c2 iter_args(%arg17 = %35, %arg18 = %36) -> (tensor<768xf32>, tensor<768xf32>) {
      %102 = arith.remui %arg16, %c48 : index
      %103 = arith.index_cast %102 : index to i64
      %104 = arith.uitofp %103 : i64 to f32
      %105 = arith.divf %104, %cst_1 : f32
      %106 = math.powf %cst_2, %105 : f32
      %107 = arith.divf %cst_0, %106 : f32
      %108 = arith.mulf %39, %107 : f32
      %109 = math.cos %108 : f32
      %110 = math.sin %108 : f32
      %111 = func.call @rot(%arg17, %arg16, %109, %110) : (tensor<768xf32>, index, f32, f32) -> tensor<768xf32>
      %112 = arith.cmpi ult, %arg16, %c768 : index
      %113 = scf.if %112 -> (tensor<768xf32>) {
        %114 = func.call @rot(%arg18, %arg16, %109, %110) : (tensor<768xf32>, index, f32, f32) -> tensor<768xf32>
        scf.yield %114 : tensor<768xf32>
      } else {
        scf.yield %arg18 : tensor<768xf32>
      }
      scf.yield %111, %113 : tensor<768xf32>, tensor<768xf32>
    }
    %inserted_slice_46 = tensor.insert_slice %40#1 into %inserted_slice_44[%33, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %extracted_slice_47 = tensor.extract_slice %inserted_slice_46[%33, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %extracted_slice_48 = tensor.extract_slice %inserted_slice_45[%33, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %41 = call @mha(%40#0, %extracted_slice_47, %extracted_slice_48, %arg1) : (tensor<768xf32>, tensor<1024x768xf32>, tensor<1024x768xf32>, index) -> tensor<768xf32>
    %42 = bufferization.materialize_in_destination %41 in %34 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_49 = tensor.extract_slice %arg9[%33, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %43 = cinm.op.gemv %extracted_slice_49, %42 plus %31 into %31 : tensor<768x768xf32>, tensor<768xf32> plus tensor<768xf32> into tensor<768xf32> -> tensor<768xf32>
    %extracted_slice_50 = tensor.extract_slice %arg13[%33, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %44 = call @rmsnorm(%43, %extracted_slice_50) : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %45 = bufferization.materialize_in_destination %44 in %42 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_51 = tensor.extract_slice %arg10[%33, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %extracted_slice_52 = tensor.extract_slice %arg12[%33, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %46 = cinm.op.gemv %extracted_slice_51, %45 : tensor<2048x768xf32>, tensor<768xf32> -> tensor<2048xf32>
    %47 = cinm.op.gemv %extracted_slice_52, %45 : tensor<2048x768xf32>, tensor<768xf32> -> tensor<2048xf32>
    %mapped_53 = linalg.map ins(%47 : tensor<2048xf32>) outs(%46 : tensor<2048xf32>)
      (%in: f32, %init: f32) {
        %102 = arith.negf %init : f32
        %103 = math.exp %102 : f32
        %104 = arith.addf %cst_0, %103 : f32
        %105 = arith.divf %cst_0, %104 : f32
        %106 = arith.mulf %init, %105 : f32
        %107 = arith.mulf %106, %in : f32
        linalg.yield %107 : f32
      }
    %extracted_slice_54 = tensor.extract_slice %arg11[%33, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
    %48 = cinm.op.gemv %extracted_slice_54, %mapped_53 plus %45 into %43 : tensor<768x2048xf32>, tensor<2048xf32> plus tensor<768xf32> into tensor<768xf32> -> tensor<768xf32>
    %c3 = arith.constant 3 : index
    %49 = arith.muli %c1, %c3 : index
    %50 = arith.addi %c0, %49 : index
    %extracted_slice_55 = tensor.extract_slice %arg5[%50, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %51 = call @rmsnorm(%48, %extracted_slice_55) : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_56 = tensor.extract_slice %arg6[%50, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_57 = tensor.extract_slice %arg7[%50, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_58 = tensor.extract_slice %arg8[%50, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_59 = tensor.extract_slice %inserted_slice_46[%50, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %extracted_slice_60 = tensor.extract_slice %inserted_slice_45[%50, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %52 = cinm.op.gemv %extracted_slice_56, %51 : tensor<768x768xf32>, tensor<768xf32> -> tensor<768xf32>
    %53 = cinm.op.gemv %extracted_slice_57, %51 : tensor<768x768xf32>, tensor<768xf32> -> tensor<768xf32>
    %54 = cinm.op.gemv %extracted_slice_58, %51 : tensor<768x768xf32>, tensor<768xf32> -> tensor<768xf32>
    %inserted_slice_61 = tensor.insert_slice %53 into %inserted_slice_46[%50, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %inserted_slice_62 = tensor.insert_slice %54 into %inserted_slice_45[%50, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %55 = arith.index_cast %arg1 : index to i64
    %56 = arith.uitofp %55 : i64 to f32
    %57:2 = scf.for %arg16 = %c0 to %c768 step %c2 iter_args(%arg17 = %52, %arg18 = %53) -> (tensor<768xf32>, tensor<768xf32>) {
      %102 = arith.remui %arg16, %c48 : index
      %103 = arith.index_cast %102 : index to i64
      %104 = arith.uitofp %103 : i64 to f32
      %105 = arith.divf %104, %cst_1 : f32
      %106 = math.powf %cst_2, %105 : f32
      %107 = arith.divf %cst_0, %106 : f32
      %108 = arith.mulf %56, %107 : f32
      %109 = math.cos %108 : f32
      %110 = math.sin %108 : f32
      %111 = func.call @rot(%arg17, %arg16, %109, %110) : (tensor<768xf32>, index, f32, f32) -> tensor<768xf32>
      %112 = arith.cmpi ult, %arg16, %c768 : index
      %113 = scf.if %112 -> (tensor<768xf32>) {
        %114 = func.call @rot(%arg18, %arg16, %109, %110) : (tensor<768xf32>, index, f32, f32) -> tensor<768xf32>
        scf.yield %114 : tensor<768xf32>
      } else {
        scf.yield %arg18 : tensor<768xf32>
      }
      scf.yield %111, %113 : tensor<768xf32>, tensor<768xf32>
    }
    %inserted_slice_63 = tensor.insert_slice %57#1 into %inserted_slice_61[%50, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %extracted_slice_64 = tensor.extract_slice %inserted_slice_63[%50, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %extracted_slice_65 = tensor.extract_slice %inserted_slice_62[%50, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %58 = call @mha(%57#0, %extracted_slice_64, %extracted_slice_65, %arg1) : (tensor<768xf32>, tensor<1024x768xf32>, tensor<1024x768xf32>, index) -> tensor<768xf32>
    %59 = bufferization.materialize_in_destination %58 in %51 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_66 = tensor.extract_slice %arg9[%50, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %60 = cinm.op.gemv %extracted_slice_66, %59 plus %48 into %48 : tensor<768x768xf32>, tensor<768xf32> plus tensor<768xf32> into tensor<768xf32> -> tensor<768xf32>
    %extracted_slice_67 = tensor.extract_slice %arg13[%50, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %61 = call @rmsnorm(%60, %extracted_slice_67) : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %62 = bufferization.materialize_in_destination %61 in %59 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_68 = tensor.extract_slice %arg10[%50, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %extracted_slice_69 = tensor.extract_slice %arg12[%50, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %63 = cinm.op.gemv %extracted_slice_68, %62 : tensor<2048x768xf32>, tensor<768xf32> -> tensor<2048xf32>
    %64 = cinm.op.gemv %extracted_slice_69, %62 : tensor<2048x768xf32>, tensor<768xf32> -> tensor<2048xf32>
    %mapped_70 = linalg.map ins(%64 : tensor<2048xf32>) outs(%63 : tensor<2048xf32>)
      (%in: f32, %init: f32) {
        %102 = arith.negf %init : f32
        %103 = math.exp %102 : f32
        %104 = arith.addf %cst_0, %103 : f32
        %105 = arith.divf %cst_0, %104 : f32
        %106 = arith.mulf %init, %105 : f32
        %107 = arith.mulf %106, %in : f32
        linalg.yield %107 : f32
      }
    %extracted_slice_71 = tensor.extract_slice %arg11[%50, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
    %65 = cinm.op.gemv %extracted_slice_71, %mapped_70 plus %62 into %60 : tensor<768x2048xf32>, tensor<2048xf32> plus tensor<768xf32> into tensor<768xf32> -> tensor<768xf32>
    %c4 = arith.constant 4 : index
    %66 = arith.muli %c1, %c4 : index
    %67 = arith.addi %c0, %66 : index
    %extracted_slice_72 = tensor.extract_slice %arg5[%67, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %68 = call @rmsnorm(%65, %extracted_slice_72) : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_73 = tensor.extract_slice %arg6[%67, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_74 = tensor.extract_slice %arg7[%67, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_75 = tensor.extract_slice %arg8[%67, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_76 = tensor.extract_slice %inserted_slice_63[%67, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %extracted_slice_77 = tensor.extract_slice %inserted_slice_62[%67, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %69 = cinm.op.gemv %extracted_slice_73, %68 : tensor<768x768xf32>, tensor<768xf32> -> tensor<768xf32>
    %70 = cinm.op.gemv %extracted_slice_74, %68 : tensor<768x768xf32>, tensor<768xf32> -> tensor<768xf32>
    %71 = cinm.op.gemv %extracted_slice_75, %68 : tensor<768x768xf32>, tensor<768xf32> -> tensor<768xf32>
    %inserted_slice_78 = tensor.insert_slice %70 into %inserted_slice_63[%67, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %inserted_slice_79 = tensor.insert_slice %71 into %inserted_slice_62[%67, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %72 = arith.index_cast %arg1 : index to i64
    %73 = arith.uitofp %72 : i64 to f32
    %74:2 = scf.for %arg16 = %c0 to %c768 step %c2 iter_args(%arg17 = %69, %arg18 = %70) -> (tensor<768xf32>, tensor<768xf32>) {
      %102 = arith.remui %arg16, %c48 : index
      %103 = arith.index_cast %102 : index to i64
      %104 = arith.uitofp %103 : i64 to f32
      %105 = arith.divf %104, %cst_1 : f32
      %106 = math.powf %cst_2, %105 : f32
      %107 = arith.divf %cst_0, %106 : f32
      %108 = arith.mulf %73, %107 : f32
      %109 = math.cos %108 : f32
      %110 = math.sin %108 : f32
      %111 = func.call @rot(%arg17, %arg16, %109, %110) : (tensor<768xf32>, index, f32, f32) -> tensor<768xf32>
      %112 = arith.cmpi ult, %arg16, %c768 : index
      %113 = scf.if %112 -> (tensor<768xf32>) {
        %114 = func.call @rot(%arg18, %arg16, %109, %110) : (tensor<768xf32>, index, f32, f32) -> tensor<768xf32>
        scf.yield %114 : tensor<768xf32>
      } else {
        scf.yield %arg18 : tensor<768xf32>
      }
      scf.yield %111, %113 : tensor<768xf32>, tensor<768xf32>
    }
    %inserted_slice_80 = tensor.insert_slice %74#1 into %inserted_slice_78[%67, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %extracted_slice_81 = tensor.extract_slice %inserted_slice_80[%67, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %extracted_slice_82 = tensor.extract_slice %inserted_slice_79[%67, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %75 = call @mha(%74#0, %extracted_slice_81, %extracted_slice_82, %arg1) : (tensor<768xf32>, tensor<1024x768xf32>, tensor<1024x768xf32>, index) -> tensor<768xf32>
    %76 = bufferization.materialize_in_destination %75 in %68 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_83 = tensor.extract_slice %arg9[%67, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %77 = cinm.op.gemv %extracted_slice_83, %76 plus %65 into %65 : tensor<768x768xf32>, tensor<768xf32> plus tensor<768xf32> into tensor<768xf32> -> tensor<768xf32>
    %extracted_slice_84 = tensor.extract_slice %arg13[%67, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %78 = call @rmsnorm(%77, %extracted_slice_84) : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %79 = bufferization.materialize_in_destination %78 in %76 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_85 = tensor.extract_slice %arg10[%67, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %extracted_slice_86 = tensor.extract_slice %arg12[%67, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %80 = cinm.op.gemv %extracted_slice_85, %79 : tensor<2048x768xf32>, tensor<768xf32> -> tensor<2048xf32>
    %81 = cinm.op.gemv %extracted_slice_86, %79 : tensor<2048x768xf32>, tensor<768xf32> -> tensor<2048xf32>
    %mapped_87 = linalg.map ins(%81 : tensor<2048xf32>) outs(%80 : tensor<2048xf32>)
      (%in: f32, %init: f32) {
        %102 = arith.negf %init : f32
        %103 = math.exp %102 : f32
        %104 = arith.addf %cst_0, %103 : f32
        %105 = arith.divf %cst_0, %104 : f32
        %106 = arith.mulf %init, %105 : f32
        %107 = arith.mulf %106, %in : f32
        linalg.yield %107 : f32
      }
    %extracted_slice_88 = tensor.extract_slice %arg11[%67, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
    %82 = cinm.op.gemv %extracted_slice_88, %mapped_87 plus %79 into %77 : tensor<768x2048xf32>, tensor<2048xf32> plus tensor<768xf32> into tensor<768xf32> -> tensor<768xf32>
    %c5 = arith.constant 5 : index
    %83 = arith.muli %c1, %c5 : index
    %84 = arith.addi %c0, %83 : index
    %extracted_slice_89 = tensor.extract_slice %arg5[%84, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %85 = call @rmsnorm(%82, %extracted_slice_89) : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_90 = tensor.extract_slice %arg6[%84, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_91 = tensor.extract_slice %arg7[%84, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_92 = tensor.extract_slice %arg8[%84, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_93 = tensor.extract_slice %inserted_slice_80[%84, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %extracted_slice_94 = tensor.extract_slice %inserted_slice_79[%84, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %86 = cinm.op.gemv %extracted_slice_90, %85 : tensor<768x768xf32>, tensor<768xf32> -> tensor<768xf32>
    %87 = cinm.op.gemv %extracted_slice_91, %85 : tensor<768x768xf32>, tensor<768xf32> -> tensor<768xf32>
    %88 = cinm.op.gemv %extracted_slice_92, %85 : tensor<768x768xf32>, tensor<768xf32> -> tensor<768xf32>
    %inserted_slice_95 = tensor.insert_slice %87 into %inserted_slice_80[%84, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %inserted_slice_96 = tensor.insert_slice %88 into %inserted_slice_79[%84, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %89 = arith.index_cast %arg1 : index to i64
    %90 = arith.uitofp %89 : i64 to f32
    %91:2 = scf.for %arg16 = %c0 to %c768 step %c2 iter_args(%arg17 = %86, %arg18 = %87) -> (tensor<768xf32>, tensor<768xf32>) {
      %102 = arith.remui %arg16, %c48 : index
      %103 = arith.index_cast %102 : index to i64
      %104 = arith.uitofp %103 : i64 to f32
      %105 = arith.divf %104, %cst_1 : f32
      %106 = math.powf %cst_2, %105 : f32
      %107 = arith.divf %cst_0, %106 : f32
      %108 = arith.mulf %90, %107 : f32
      %109 = math.cos %108 : f32
      %110 = math.sin %108 : f32
      %111 = func.call @rot(%arg17, %arg16, %109, %110) : (tensor<768xf32>, index, f32, f32) -> tensor<768xf32>
      %112 = arith.cmpi ult, %arg16, %c768 : index
      %113 = scf.if %112 -> (tensor<768xf32>) {
        %114 = func.call @rot(%arg18, %arg16, %109, %110) : (tensor<768xf32>, index, f32, f32) -> tensor<768xf32>
        scf.yield %114 : tensor<768xf32>
      } else {
        scf.yield %arg18 : tensor<768xf32>
      }
      scf.yield %111, %113 : tensor<768xf32>, tensor<768xf32>
    }
    %inserted_slice_97 = tensor.insert_slice %91#1 into %inserted_slice_95[%84, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %extracted_slice_98 = tensor.extract_slice %inserted_slice_97[%84, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %extracted_slice_99 = tensor.extract_slice %inserted_slice_96[%84, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %92 = call @mha(%91#0, %extracted_slice_98, %extracted_slice_99, %arg1) : (tensor<768xf32>, tensor<1024x768xf32>, tensor<1024x768xf32>, index) -> tensor<768xf32>
    %93 = bufferization.materialize_in_destination %92 in %85 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_100 = tensor.extract_slice %arg9[%84, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %94 = cinm.op.gemv %extracted_slice_100, %93 plus %82 into %82 : tensor<768x768xf32>, tensor<768xf32> plus tensor<768xf32> into tensor<768xf32> -> tensor<768xf32>
    %extracted_slice_101 = tensor.extract_slice %arg13[%84, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %95 = call @rmsnorm(%94, %extracted_slice_101) : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %96 = bufferization.materialize_in_destination %95 in %93 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_102 = tensor.extract_slice %arg10[%84, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %extracted_slice_103 = tensor.extract_slice %arg12[%84, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %97 = cinm.op.gemv %extracted_slice_102, %96 : tensor<2048x768xf32>, tensor<768xf32> -> tensor<2048xf32>
    %98 = cinm.op.gemv %extracted_slice_103, %96 : tensor<2048x768xf32>, tensor<768xf32> -> tensor<2048xf32>
    %mapped_104 = linalg.map ins(%98 : tensor<2048xf32>) outs(%97 : tensor<2048xf32>)
      (%in: f32, %init: f32) {
        %102 = arith.negf %init : f32
        %103 = math.exp %102 : f32
        %104 = arith.addf %cst_0, %103 : f32
        %105 = arith.divf %cst_0, %104 : f32
        %106 = arith.mulf %init, %105 : f32
        %107 = arith.mulf %106, %in : f32
        linalg.yield %107 : f32
      }
    %extracted_slice_105 = tensor.extract_slice %arg11[%84, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
    %99 = cinm.op.gemv %extracted_slice_105, %mapped_104 plus %96 into %94 : tensor<768x2048xf32>, tensor<2048xf32> plus tensor<768xf32> into tensor<768xf32> -> tensor<768xf32>
    %100 = call @rmsnorm(%99, %arg14) : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %padded = tensor.pad %arg15 low[0, 0] high[2048, 0] {
    ^bb0(%arg16: index, %arg17: index):
      tensor.yield %cst : f32
    } : tensor<32000x768xf32> to tensor<34048x768xf32>
    %101 = cinm.op.gemv %padded, %100 : tensor<34048x768xf32>, tensor<768xf32> -> tensor<34048xf32>
    %extracted_slice_106 = tensor.extract_slice %101[0] [32000] [1] : tensor<34048xf32> to tensor<32000xf32>
    return %extracted_slice_106 : tensor<32000xf32>
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
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c48 = arith.constant 48 : index
    %c768 = arith.constant 768 : index
    %c1024 = arith.constant 1024 : index
    %cst = arith.constant 6.92820311 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %0 = arith.addi %arg3, %c1 : index
    %1 = tensor.empty() : tensor<768xf32>
    %c768_1 = arith.constant 768 : index
    %extracted_slice = tensor.extract_slice %arg0[%c0] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_2 = tensor.extract_slice %arg1[0, %c0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %2 = cinm.op.gemv %extracted_slice_2, %extracted_slice : tensor<1024x48xf32>, tensor<48xf32> -> tensor<1024xf32>
    %splat = tensor.splat %cst : tensor<1024xf32>
    %3 = cinm.op.elementwise div %2, %splat : tensor<1024xf32>
    %4 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %3) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_0 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %5 = call @softmax(%4) : (tensor<1024xf32>) -> tensor<1024xf32>
    %extracted_slice_3 = tensor.extract_slice %arg2[0, %c0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %expanded = tensor.expand_shape %5 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %6 = cinm.op.gemm %expanded, %extracted_slice_3 : tensor<1x1024xf32>, tensor<1024x48xf32> -> tensor<1x48xf32>
    %collapsed = tensor.collapse_shape %6 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice = tensor.insert_slice %collapsed into %1[%c0] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %c1_4 = arith.constant 1 : index
    %7 = arith.muli %c48, %c1_4 : index
    %8 = arith.addi %c0, %7 : index
    %extracted_slice_5 = tensor.extract_slice %arg0[%8] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_6 = tensor.extract_slice %arg1[0, %8] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %9 = cinm.op.gemv %extracted_slice_6, %extracted_slice_5 : tensor<1024x48xf32>, tensor<48xf32> -> tensor<1024xf32>
    %splat_7 = tensor.splat %cst : tensor<1024xf32>
    %10 = cinm.op.elementwise div %9, %splat_7 : tensor<1024xf32>
    %11 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %10) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_0 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %12 = call @softmax(%11) : (tensor<1024xf32>) -> tensor<1024xf32>
    %extracted_slice_8 = tensor.extract_slice %arg2[0, %8] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %expanded_9 = tensor.expand_shape %12 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %13 = cinm.op.gemm %expanded_9, %extracted_slice_8 : tensor<1x1024xf32>, tensor<1024x48xf32> -> tensor<1x48xf32>
    %collapsed_10 = tensor.collapse_shape %13 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_11 = tensor.insert_slice %collapsed_10 into %inserted_slice[%8] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %c2 = arith.constant 2 : index
    %14 = arith.muli %c48, %c2 : index
    %15 = arith.addi %c0, %14 : index
    %extracted_slice_12 = tensor.extract_slice %arg0[%15] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_13 = tensor.extract_slice %arg1[0, %15] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %16 = cinm.op.gemv %extracted_slice_13, %extracted_slice_12 : tensor<1024x48xf32>, tensor<48xf32> -> tensor<1024xf32>
    %splat_14 = tensor.splat %cst : tensor<1024xf32>
    %17 = cinm.op.elementwise div %16, %splat_14 : tensor<1024xf32>
    %18 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %17) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_0 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %19 = call @softmax(%18) : (tensor<1024xf32>) -> tensor<1024xf32>
    %extracted_slice_15 = tensor.extract_slice %arg2[0, %15] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %expanded_16 = tensor.expand_shape %19 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %20 = cinm.op.gemm %expanded_16, %extracted_slice_15 : tensor<1x1024xf32>, tensor<1024x48xf32> -> tensor<1x48xf32>
    %collapsed_17 = tensor.collapse_shape %20 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_18 = tensor.insert_slice %collapsed_17 into %inserted_slice_11[%15] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %c3 = arith.constant 3 : index
    %21 = arith.muli %c48, %c3 : index
    %22 = arith.addi %c0, %21 : index
    %extracted_slice_19 = tensor.extract_slice %arg0[%22] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_20 = tensor.extract_slice %arg1[0, %22] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %23 = cinm.op.gemv %extracted_slice_20, %extracted_slice_19 : tensor<1024x48xf32>, tensor<48xf32> -> tensor<1024xf32>
    %splat_21 = tensor.splat %cst : tensor<1024xf32>
    %24 = cinm.op.elementwise div %23, %splat_21 : tensor<1024xf32>
    %25 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %24) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_0 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %26 = call @softmax(%25) : (tensor<1024xf32>) -> tensor<1024xf32>
    %extracted_slice_22 = tensor.extract_slice %arg2[0, %22] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %expanded_23 = tensor.expand_shape %26 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %27 = cinm.op.gemm %expanded_23, %extracted_slice_22 : tensor<1x1024xf32>, tensor<1024x48xf32> -> tensor<1x48xf32>
    %collapsed_24 = tensor.collapse_shape %27 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_25 = tensor.insert_slice %collapsed_24 into %inserted_slice_18[%22] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %c4 = arith.constant 4 : index
    %28 = arith.muli %c48, %c4 : index
    %29 = arith.addi %c0, %28 : index
    %extracted_slice_26 = tensor.extract_slice %arg0[%29] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_27 = tensor.extract_slice %arg1[0, %29] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %30 = cinm.op.gemv %extracted_slice_27, %extracted_slice_26 : tensor<1024x48xf32>, tensor<48xf32> -> tensor<1024xf32>
    %splat_28 = tensor.splat %cst : tensor<1024xf32>
    %31 = cinm.op.elementwise div %30, %splat_28 : tensor<1024xf32>
    %32 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %31) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_0 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %33 = call @softmax(%32) : (tensor<1024xf32>) -> tensor<1024xf32>
    %extracted_slice_29 = tensor.extract_slice %arg2[0, %29] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %expanded_30 = tensor.expand_shape %33 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %34 = cinm.op.gemm %expanded_30, %extracted_slice_29 : tensor<1x1024xf32>, tensor<1024x48xf32> -> tensor<1x48xf32>
    %collapsed_31 = tensor.collapse_shape %34 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_32 = tensor.insert_slice %collapsed_31 into %inserted_slice_25[%29] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %c5 = arith.constant 5 : index
    %35 = arith.muli %c48, %c5 : index
    %36 = arith.addi %c0, %35 : index
    %extracted_slice_33 = tensor.extract_slice %arg0[%36] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_34 = tensor.extract_slice %arg1[0, %36] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %37 = cinm.op.gemv %extracted_slice_34, %extracted_slice_33 : tensor<1024x48xf32>, tensor<48xf32> -> tensor<1024xf32>
    %splat_35 = tensor.splat %cst : tensor<1024xf32>
    %38 = cinm.op.elementwise div %37, %splat_35 : tensor<1024xf32>
    %39 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %38) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_0 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %40 = call @softmax(%39) : (tensor<1024xf32>) -> tensor<1024xf32>
    %extracted_slice_36 = tensor.extract_slice %arg2[0, %36] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %expanded_37 = tensor.expand_shape %40 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %41 = cinm.op.gemm %expanded_37, %extracted_slice_36 : tensor<1x1024xf32>, tensor<1024x48xf32> -> tensor<1x48xf32>
    %collapsed_38 = tensor.collapse_shape %41 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_39 = tensor.insert_slice %collapsed_38 into %inserted_slice_32[%36] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %c6 = arith.constant 6 : index
    %42 = arith.muli %c48, %c6 : index
    %43 = arith.addi %c0, %42 : index
    %extracted_slice_40 = tensor.extract_slice %arg0[%43] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_41 = tensor.extract_slice %arg1[0, %43] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %44 = cinm.op.gemv %extracted_slice_41, %extracted_slice_40 : tensor<1024x48xf32>, tensor<48xf32> -> tensor<1024xf32>
    %splat_42 = tensor.splat %cst : tensor<1024xf32>
    %45 = cinm.op.elementwise div %44, %splat_42 : tensor<1024xf32>
    %46 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %45) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_0 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %47 = call @softmax(%46) : (tensor<1024xf32>) -> tensor<1024xf32>
    %extracted_slice_43 = tensor.extract_slice %arg2[0, %43] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %expanded_44 = tensor.expand_shape %47 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %48 = cinm.op.gemm %expanded_44, %extracted_slice_43 : tensor<1x1024xf32>, tensor<1024x48xf32> -> tensor<1x48xf32>
    %collapsed_45 = tensor.collapse_shape %48 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_46 = tensor.insert_slice %collapsed_45 into %inserted_slice_39[%43] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %c7 = arith.constant 7 : index
    %49 = arith.muli %c48, %c7 : index
    %50 = arith.addi %c0, %49 : index
    %extracted_slice_47 = tensor.extract_slice %arg0[%50] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_48 = tensor.extract_slice %arg1[0, %50] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %51 = cinm.op.gemv %extracted_slice_48, %extracted_slice_47 : tensor<1024x48xf32>, tensor<48xf32> -> tensor<1024xf32>
    %splat_49 = tensor.splat %cst : tensor<1024xf32>
    %52 = cinm.op.elementwise div %51, %splat_49 : tensor<1024xf32>
    %53 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %52) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_0 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %54 = call @softmax(%53) : (tensor<1024xf32>) -> tensor<1024xf32>
    %extracted_slice_50 = tensor.extract_slice %arg2[0, %50] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %expanded_51 = tensor.expand_shape %54 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %55 = cinm.op.gemm %expanded_51, %extracted_slice_50 : tensor<1x1024xf32>, tensor<1024x48xf32> -> tensor<1x48xf32>
    %collapsed_52 = tensor.collapse_shape %55 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_53 = tensor.insert_slice %collapsed_52 into %inserted_slice_46[%50] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %c8 = arith.constant 8 : index
    %56 = arith.muli %c48, %c8 : index
    %57 = arith.addi %c0, %56 : index
    %extracted_slice_54 = tensor.extract_slice %arg0[%57] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_55 = tensor.extract_slice %arg1[0, %57] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %58 = cinm.op.gemv %extracted_slice_55, %extracted_slice_54 : tensor<1024x48xf32>, tensor<48xf32> -> tensor<1024xf32>
    %splat_56 = tensor.splat %cst : tensor<1024xf32>
    %59 = cinm.op.elementwise div %58, %splat_56 : tensor<1024xf32>
    %60 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %59) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_0 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %61 = call @softmax(%60) : (tensor<1024xf32>) -> tensor<1024xf32>
    %extracted_slice_57 = tensor.extract_slice %arg2[0, %57] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %expanded_58 = tensor.expand_shape %61 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %62 = cinm.op.gemm %expanded_58, %extracted_slice_57 : tensor<1x1024xf32>, tensor<1024x48xf32> -> tensor<1x48xf32>
    %collapsed_59 = tensor.collapse_shape %62 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_60 = tensor.insert_slice %collapsed_59 into %inserted_slice_53[%57] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %c9 = arith.constant 9 : index
    %63 = arith.muli %c48, %c9 : index
    %64 = arith.addi %c0, %63 : index
    %extracted_slice_61 = tensor.extract_slice %arg0[%64] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_62 = tensor.extract_slice %arg1[0, %64] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %65 = cinm.op.gemv %extracted_slice_62, %extracted_slice_61 : tensor<1024x48xf32>, tensor<48xf32> -> tensor<1024xf32>
    %splat_63 = tensor.splat %cst : tensor<1024xf32>
    %66 = cinm.op.elementwise div %65, %splat_63 : tensor<1024xf32>
    %67 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %66) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_0 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %68 = call @softmax(%67) : (tensor<1024xf32>) -> tensor<1024xf32>
    %extracted_slice_64 = tensor.extract_slice %arg2[0, %64] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %expanded_65 = tensor.expand_shape %68 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %69 = cinm.op.gemm %expanded_65, %extracted_slice_64 : tensor<1x1024xf32>, tensor<1024x48xf32> -> tensor<1x48xf32>
    %collapsed_66 = tensor.collapse_shape %69 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_67 = tensor.insert_slice %collapsed_66 into %inserted_slice_60[%64] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %c10 = arith.constant 10 : index
    %70 = arith.muli %c48, %c10 : index
    %71 = arith.addi %c0, %70 : index
    %extracted_slice_68 = tensor.extract_slice %arg0[%71] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_69 = tensor.extract_slice %arg1[0, %71] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %72 = cinm.op.gemv %extracted_slice_69, %extracted_slice_68 : tensor<1024x48xf32>, tensor<48xf32> -> tensor<1024xf32>
    %splat_70 = tensor.splat %cst : tensor<1024xf32>
    %73 = cinm.op.elementwise div %72, %splat_70 : tensor<1024xf32>
    %74 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %73) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_0 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %75 = call @softmax(%74) : (tensor<1024xf32>) -> tensor<1024xf32>
    %extracted_slice_71 = tensor.extract_slice %arg2[0, %71] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %expanded_72 = tensor.expand_shape %75 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %76 = cinm.op.gemm %expanded_72, %extracted_slice_71 : tensor<1x1024xf32>, tensor<1024x48xf32> -> tensor<1x48xf32>
    %collapsed_73 = tensor.collapse_shape %76 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_74 = tensor.insert_slice %collapsed_73 into %inserted_slice_67[%71] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %c11 = arith.constant 11 : index
    %77 = arith.muli %c48, %c11 : index
    %78 = arith.addi %c0, %77 : index
    %extracted_slice_75 = tensor.extract_slice %arg0[%78] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_76 = tensor.extract_slice %arg1[0, %78] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %79 = cinm.op.gemv %extracted_slice_76, %extracted_slice_75 : tensor<1024x48xf32>, tensor<48xf32> -> tensor<1024xf32>
    %splat_77 = tensor.splat %cst : tensor<1024xf32>
    %80 = cinm.op.elementwise div %79, %splat_77 : tensor<1024xf32>
    %81 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %80) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_0 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %82 = call @softmax(%81) : (tensor<1024xf32>) -> tensor<1024xf32>
    %extracted_slice_78 = tensor.extract_slice %arg2[0, %78] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %expanded_79 = tensor.expand_shape %82 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %83 = cinm.op.gemm %expanded_79, %extracted_slice_78 : tensor<1x1024xf32>, tensor<1024x48xf32> -> tensor<1x48xf32>
    %collapsed_80 = tensor.collapse_shape %83 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_81 = tensor.insert_slice %collapsed_80 into %inserted_slice_74[%78] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %c12 = arith.constant 12 : index
    %84 = arith.muli %c48, %c12 : index
    %85 = arith.addi %c0, %84 : index
    %extracted_slice_82 = tensor.extract_slice %arg0[%85] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_83 = tensor.extract_slice %arg1[0, %85] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %86 = cinm.op.gemv %extracted_slice_83, %extracted_slice_82 : tensor<1024x48xf32>, tensor<48xf32> -> tensor<1024xf32>
    %splat_84 = tensor.splat %cst : tensor<1024xf32>
    %87 = cinm.op.elementwise div %86, %splat_84 : tensor<1024xf32>
    %88 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %87) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_0 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %89 = call @softmax(%88) : (tensor<1024xf32>) -> tensor<1024xf32>
    %extracted_slice_85 = tensor.extract_slice %arg2[0, %85] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %expanded_86 = tensor.expand_shape %89 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %90 = cinm.op.gemm %expanded_86, %extracted_slice_85 : tensor<1x1024xf32>, tensor<1024x48xf32> -> tensor<1x48xf32>
    %collapsed_87 = tensor.collapse_shape %90 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_88 = tensor.insert_slice %collapsed_87 into %inserted_slice_81[%85] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %c13 = arith.constant 13 : index
    %91 = arith.muli %c48, %c13 : index
    %92 = arith.addi %c0, %91 : index
    %extracted_slice_89 = tensor.extract_slice %arg0[%92] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_90 = tensor.extract_slice %arg1[0, %92] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %93 = cinm.op.gemv %extracted_slice_90, %extracted_slice_89 : tensor<1024x48xf32>, tensor<48xf32> -> tensor<1024xf32>
    %splat_91 = tensor.splat %cst : tensor<1024xf32>
    %94 = cinm.op.elementwise div %93, %splat_91 : tensor<1024xf32>
    %95 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %94) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_0 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %96 = call @softmax(%95) : (tensor<1024xf32>) -> tensor<1024xf32>
    %extracted_slice_92 = tensor.extract_slice %arg2[0, %92] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %expanded_93 = tensor.expand_shape %96 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %97 = cinm.op.gemm %expanded_93, %extracted_slice_92 : tensor<1x1024xf32>, tensor<1024x48xf32> -> tensor<1x48xf32>
    %collapsed_94 = tensor.collapse_shape %97 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_95 = tensor.insert_slice %collapsed_94 into %inserted_slice_88[%92] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %c14 = arith.constant 14 : index
    %98 = arith.muli %c48, %c14 : index
    %99 = arith.addi %c0, %98 : index
    %extracted_slice_96 = tensor.extract_slice %arg0[%99] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_97 = tensor.extract_slice %arg1[0, %99] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %100 = cinm.op.gemv %extracted_slice_97, %extracted_slice_96 : tensor<1024x48xf32>, tensor<48xf32> -> tensor<1024xf32>
    %splat_98 = tensor.splat %cst : tensor<1024xf32>
    %101 = cinm.op.elementwise div %100, %splat_98 : tensor<1024xf32>
    %102 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %101) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_0 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %103 = call @softmax(%102) : (tensor<1024xf32>) -> tensor<1024xf32>
    %extracted_slice_99 = tensor.extract_slice %arg2[0, %99] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %expanded_100 = tensor.expand_shape %103 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %104 = cinm.op.gemm %expanded_100, %extracted_slice_99 : tensor<1x1024xf32>, tensor<1024x48xf32> -> tensor<1x48xf32>
    %collapsed_101 = tensor.collapse_shape %104 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_102 = tensor.insert_slice %collapsed_101 into %inserted_slice_95[%99] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %c15 = arith.constant 15 : index
    %105 = arith.muli %c48, %c15 : index
    %106 = arith.addi %c0, %105 : index
    %extracted_slice_103 = tensor.extract_slice %arg0[%106] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_104 = tensor.extract_slice %arg1[0, %106] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %107 = cinm.op.gemv %extracted_slice_104, %extracted_slice_103 : tensor<1024x48xf32>, tensor<48xf32> -> tensor<1024xf32>
    %splat_105 = tensor.splat %cst : tensor<1024xf32>
    %108 = cinm.op.elementwise div %107, %splat_105 : tensor<1024xf32>
    %109 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %108) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_0 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %110 = call @softmax(%109) : (tensor<1024xf32>) -> tensor<1024xf32>
    %extracted_slice_106 = tensor.extract_slice %arg2[0, %106] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %expanded_107 = tensor.expand_shape %110 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %111 = cinm.op.gemm %expanded_107, %extracted_slice_106 : tensor<1x1024xf32>, tensor<1024x48xf32> -> tensor<1x48xf32>
    %collapsed_108 = tensor.collapse_shape %111 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_109 = tensor.insert_slice %collapsed_108 into %inserted_slice_102[%106] [48] [1] : tensor<48xf32> into tensor<768xf32>
    return %inserted_slice_109 : tensor<768xf32>
  }
  func.func @rmsnorm(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>) -> tensor<768xf32> {
    %cst = arith.constant 9.99999974E-6 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 7.680000e+02 : f32
    %0 = cinm.op.elementwise mul %arg0, %arg0 : tensor<768xf32>
    %1 = cinm.op.reduce add(%0) : tensor<768xf32> -> f32
    %2 = arith.divf %1, %cst_1 : f32
    %3 = arith.addf %2, %cst : f32
    %4 = math.rsqrt %3 : f32
    %splat = tensor.splat %4 : tensor<768xf32>
    %5 = cinm.op.elementwise mul %arg0, %splat : tensor<768xf32>
    %6 = cinm.op.elementwise mul %5, %arg1 : tensor<768xf32>
    return %6 : tensor<768xf32>
  }
  func.func @softmax(%arg0: tensor<1024xf32> {bufferization.writable = true}) -> tensor<1024xf32> {
    %0 = cinm.op.reduce maxnumf(%arg0) : tensor<1024xf32> -> f32
    %splat = tensor.splat %0 : tensor<1024xf32>
    %1 = cinm.op.elementwise sub %arg0, %splat into %arg0 : tensor<1024xf32> into tensor<1024xf32>
    %2 = cinm.op.elementwise exp %1 into %arg0 : tensor<1024xf32> into tensor<1024xf32>
    %3 = cinm.op.reduce add(%2) : tensor<1024xf32> -> f32
    %splat_0 = tensor.splat %3 : tensor<1024xf32>
    %4 = cinm.op.elementwise div %2, %splat_0 into %arg0 : tensor<1024xf32> into tensor<1024xf32>
    return %4 : tensor<1024xf32>
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
