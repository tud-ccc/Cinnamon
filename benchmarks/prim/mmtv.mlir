//#! cinm-opt --cinm-assign-platforms --cinm-isolate-compute-blocks --upmem-infer-accelerator --split-input-file

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

// %A is the weight operand: same data on every inference, so its
// transfer and any repack of it amortize over the serving lifetime
// (cinm.static, see cinm::isStaticValue). %x is the per-inference input.
func.func @mmtv_4MB(%A: tensor<32x64x512xi32> {cinm.static}, %x: tensor<32x512xi32>) -> tensor<32x64xi32>
  attributes {cinm.available_platforms = [#upmem]} {
  %4 = cinm.op.batch_gemv %A, %x : tensor<32x64x512xi32>, tensor<32x512xi32> -> tensor<32x64xi32>
  return %4 : tensor<32x64xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @mmtv_64MB(%A: tensor<128x256x512xi32> {cinm.static}, %x: tensor<128x512xi32>) -> tensor<128x256xi32>
  attributes {cinm.available_platforms = [#upmem]} {
  %4 = cinm.op.batch_gemv %A, %x : tensor<128x256x512xi32>, tensor<128x512xi32> -> tensor<128x256xi32>
  return %4 : tensor<128x256xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @mmtv_256MB(%A: tensor<256x512x512xi32> {cinm.static}, %x: tensor<256x512xi32>) -> tensor<256x512xi32>
  attributes {cinm.available_platforms = [#upmem]} {
  %4 = cinm.op.batch_gemv %A, %x : tensor<256x512x512xi32>, tensor<256x512xi32> -> tensor<256x512xi32>
  return %4 : tensor<256x512xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @mmtv_512MB(%A: tensor<512x512x512xi32> {cinm.static}, %x: tensor<512x512xi32>) -> tensor<512x512xi32>
  attributes {cinm.available_platforms = [#upmem]} {
  %4 = cinm.op.batch_gemv %A, %x : tensor<512x512x512xi32>, tensor<512x512xi32> -> tensor<512x512xi32>
  return %4 : tensor<512x512xi32>
}
