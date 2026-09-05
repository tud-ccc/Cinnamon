//#! cinm-opt --cinm-assign-platforms --cinm-isolate-compute-blocks --upmem-infer-accelerator --split-input-file

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

// %A is the weight operand: same data on every inference, so its
// transfer and any repack of it amortize over the serving lifetime
// (cinm.static, see cinm::isStaticValue). %x is the per-inference input.
func.func @mtv_4MB(%A: tensor<1024x1024xi32> {cinm.static}, %x: tensor<1024xi32>) -> tensor<1024xi32>
  attributes {cinm.available_platforms = [#upmem]} {
  %4 = cinm.op.gemv %A, %x : tensor<1024x1024xi32>, tensor<1024xi32> -> tensor<1024xi32>
  return %4 : tensor<1024xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @mtv_64MB(%A: tensor<4096x4096xi32> {cinm.static}, %x: tensor<4096xi32>) -> tensor<4096xi32>
  attributes {cinm.available_platforms = [#upmem]} {
  %4 = cinm.op.gemv %A, %x : tensor<4096x4096xi32>, tensor<4096xi32> -> tensor<4096xi32>
  return %4 : tensor<4096xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @mtv_256MB(%A: tensor<8192x8192xi32> {cinm.static}, %x: tensor<8192xi32>) -> tensor<8192xi32>
  attributes {cinm.available_platforms = [#upmem]} {
  %4 = cinm.op.gemv %A, %x : tensor<8192x8192xi32>, tensor<8192xi32> -> tensor<8192xi32>
  return %4 : tensor<8192xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @mtv_512MB(%A: tensor<8192x16384xi32> {cinm.static}, %x: tensor<16384xi32>) -> tensor<8192xi32>
  attributes {cinm.available_platforms = [#upmem]} {
  %4 = cinm.op.gemv %A, %x : tensor<8192x16384xi32>, tensor<16384xi32> -> tensor<8192xi32>
  return %4 : tensor<8192xi32>
}
