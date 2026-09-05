//#! cinm-opt --cinm-assign-platforms --cinm-isolate-compute-blocks --upmem-infer-accelerator --split-input-file

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @va_4MB(%x: tensor<1048576xi32>, %y: tensor<1048576xi32>) -> tensor<1048576xi32>
  attributes {cinm.available_platforms = [#upmem]} {
  %4 = cinm.op.elementwise add %x, %y : tensor<1048576xi32>
  return %4 : tensor<1048576xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @va_64MB(%x: tensor<16777216xi32>, %y: tensor<16777216xi32>) -> tensor<16777216xi32>
  attributes {cinm.available_platforms = [#upmem]} {
  %4 = cinm.op.elementwise add %x, %y : tensor<16777216xi32>
  return %4 : tensor<16777216xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @va_256MB(%x: tensor<67108864xi32>, %y: tensor<67108864xi32>) -> tensor<67108864xi32>
  attributes {cinm.available_platforms = [#upmem]} {
  %4 = cinm.op.elementwise add %x, %y : tensor<67108864xi32>
  return %4 : tensor<67108864xi32>
}
