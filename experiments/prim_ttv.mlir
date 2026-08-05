//#! cinm-opt --cinm-assign-platforms --cinm-isolate-compute-blocks --upmem-infer-accelerator --split-input-file

// -----
#upmem = #upmem.platform<type = v1A, dimensions = 32x64x24>

func.func @ttv_4MB(%A: tensor<32x64x512xi32>, %x: tensor<512xi32>) -> tensor<32x64xi32>
  attributes {cinm.available_platforms = [#upmem]} {
  %4 = cinm.compute -> tensor<32x64xi32> attributes {cinm.available_platforms = [#upmem]}{
    %e = tensor.empty(): tensor<32x512xi32>
    %xs = linalg.generic {iterator_types=["parallel", "parallel"], indexing_maps=[affine_map<(i,j)->(j)>, affine_map<(i,j)->(i,j)>]}
    ins(%x: tensor<512xi32>) outs(%e: tensor<32x512xi32>) {
      ^bb0(%xi: i32, %eij: i32):
        linalg.yield %xi : i32
    } -> tensor<32x512xi32>
    %4 = cinm.op.batch_gemv %A, %xs : tensor<32x64x512xi32>, tensor<32x512xi32> -> tensor<32x64xi32>
    cinm.yield %4 : tensor<32x64xi32>
  }
  return %4 : tensor<32x64xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dimensions = 32x64x24>

func.func @ttv_64MB(%A: tensor<128x256x512xi32>, %x: tensor<512xi32>) -> tensor<128x256xi32>
  attributes {cinm.available_platforms = [#upmem]} {
  %4 = cinm.compute -> tensor<128x256xi32> attributes {cinm.available_platforms = [#upmem]}{
    %e = tensor.empty(): tensor<128x512xi32>
    %xs = linalg.generic {iterator_types=["parallel", "parallel"], indexing_maps=[affine_map<(i,j)->(j)>, affine_map<(i,j)->(i,j)>]}
    ins(%x: tensor<512xi32>) outs(%e: tensor<128x512xi32>) {
      ^bb0(%xi: i32, %eij: i32):
        linalg.yield %xi : i32
    } -> tensor<128x512xi32>
    %4 = cinm.op.batch_gemv %A, %xs : tensor<128x256x512xi32>, tensor<128x512xi32> -> tensor<128x256xi32>
    cinm.yield %4 : tensor<128x256xi32>
  }
  return %4 : tensor<128x256xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dimensions = 32x64x24>

func.func @ttv_256MB(%A: tensor<256x512x512xi32>, %x: tensor<512xi32>) -> tensor<256x512xi32>
  attributes {cinm.available_platforms = [#upmem]} {
  %4 = cinm.compute -> tensor<256x512xi32> attributes {cinm.available_platforms = [#upmem]}{
    %e = tensor.empty(): tensor<256x512xi32>
    %xs = linalg.generic {iterator_types=["parallel", "parallel"], indexing_maps=[affine_map<(i,j)->(j)>, affine_map<(i,j)->(i,j)>]}
    ins(%x: tensor<512xi32>) outs(%e: tensor<256x512xi32>) {
      ^bb0(%xi: i32, %eij: i32):
        linalg.yield %xi : i32
    } -> tensor<256x512xi32>
    %4 = cinm.op.batch_gemv %A, %xs : tensor<256x512x512xi32>, tensor<256x512xi32> -> tensor<256x512xi32>
    cinm.yield %4 : tensor<256x512xi32>
  }
  return %4 : tensor<256x512xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dimensions = 32x64x24>

func.func @ttv_512MB(%A: tensor<512x512x512xi32>, %x: tensor<512xi32>) -> tensor<512x512xi32>
  attributes {cinm.available_platforms = [#upmem]} {
  %4 = cinm.compute -> tensor<512x512xi32> attributes {cinm.available_platforms = [#upmem]}{
    %e = tensor.empty(): tensor<512x512xi32>
    %xs = linalg.generic {iterator_types=["parallel", "parallel"], indexing_maps=[affine_map<(i,j)->(j)>, affine_map<(i,j)->(i,j)>]}
    ins(%x: tensor<512xi32>) outs(%e: tensor<512x512xi32>) {
      ^bb0(%xi: i32, %eij: i32):
        linalg.yield %xi : i32
    } -> tensor<512x512xi32>
    %4 = cinm.op.batch_gemv %A, %xs : tensor<512x512x512xi32>, tensor<512x512xi32> -> tensor<512x512xi32>
    cinm.yield %4 : tensor<512x512xi32>
  }
  return %4 : tensor<512x512xi32>
}
