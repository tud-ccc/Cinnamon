//#! cinm-opt --cinm-assign-platforms --cinm-isolate-compute-blocks --upmem-infer-accelerator --split-input-file

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @mtv_4MB(%A: tensor<1024x1024xi32>, %x: tensor<1024xi32>) -> tensor<1024xi32>
  attributes {cinm.available_platforms = [#upmem]} {
  %4 = cinm.op.gemv %A, %x : tensor<1024x1024xi32>, tensor<1024xi32> -> tensor<1024xi32>
  return %4 : tensor<1024xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @mtv_64MB(%A: tensor<4096x4096xi32>, %x: tensor<4096xi32>) -> tensor<4096xi32>
  attributes {cinm.available_platforms = [#upmem]} {
  %4 = cinm.op.gemv %A, %x : tensor<4096x4096xi32>, tensor<4096xi32> -> tensor<4096xi32>
  return %4 : tensor<4096xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @mtv_256MB(%A: tensor<8192x8192xi32>, %x: tensor<8192xi32>) -> tensor<8192xi32>
  attributes {cinm.available_platforms = [#upmem]} {
  %4 = cinm.op.gemv %A, %x : tensor<8192x8192xi32>, tensor<8192xi32> -> tensor<8192xi32>
  return %4 : tensor<8192xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @mtv_512MB(%A: tensor<8192x16384xi32>, %x: tensor<16384xi32>) -> tensor<8192xi32>
  attributes {cinm.available_platforms = [#upmem]} {
  %4 = cinm.op.gemv %A, %x : tensor<8192x16384xi32>, tensor<16384xi32> -> tensor<8192xi32>
  return %4 : tensor<8192xi32>
}
