//#! cinm-opt --cinm-assign-platforms --cinm-isolate-compute-blocks --upmem-infer-accelerator --split-input-file

#upmem = #upmem.platform<type = v1A, dimensions = 1x1x24>

func.func @gemv_dynamic(%A: tensor<?x?xf32>, %x: tensor<?xf32>) -> tensor<?xf32> 
  attributes {cinm.available_platforms = [#upmem]} {
  %4 = cinm.op.gemv %A, %x : tensor<?x?xf32>, tensor<?xf32> -> tensor<?xf32>
  return %4 : tensor<?xf32>
}

// -----
#upmem = #upmem.platform<type = v1A, dimensions = 1x1x24>

func.func @gemv_larger(%A: tensor<2048x768xf32>, %x: tensor<768xf32>) -> tensor<2048xf32> 
  attributes {cinm.available_platforms = [#upmem]} {
  %4 = cinm.op.gemv %A, %x : tensor<2048x768xf32>, tensor<768xf32> -> tensor<2048xf32>
  return %4 : tensor<2048xf32>
}

// -----
#upmem = #upmem.platform<type = v1A, dimensions = 1x1x24>

func.func @gemv_square(%A: tensor<768x768xf32>, %x: tensor<768xf32>) -> tensor<768xf32>
  attributes {cinm.available_platforms = [#upmem]} {
  %4 = cinm.op.gemv %A, %x : tensor<768x768xf32>, tensor<768xf32> -> tensor<768xf32>
  return %4 : tensor<768xf32>
}

