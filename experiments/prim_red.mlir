//#! cinm-opt --cinm-assign-platforms --cinm-isolate-compute-blocks --upmem-infer-accelerator --split-input-file


// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @red_4MB(%x: tensor<524288xi64> {bufferization.buffer_layout = affine_map<(d0) -> (d0)>, bufferization.writable = true}) -> i64
  attributes {cinm.available_platforms = [#upmem]} {
  %4 = cinm.op.reduce add (%x) : tensor<524288xi64> -> i64
  return %4 : i64
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @red_64MB(%x: tensor<8388608xi64>{bufferization.buffer_layout = affine_map<(d0) -> (d0)>, bufferization.writable = true}) -> i64
  attributes {cinm.available_platforms = [#upmem]} {
  %4 = cinm.op.reduce add (%x) : tensor<8388608xi64> -> i64
  return %4 : i64
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @red_256MB(%x: tensor<33554432xi64>{bufferization.buffer_layout = affine_map<(d0) -> (d0)>, bufferization.writable = true}) -> i64
  attributes {cinm.available_platforms = [#upmem]} {
  %4 = cinm.op.reduce add (%x) : tensor<33554432xi64> -> i64
  return %4 : i64
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @red_512MB(%x: tensor<67108864xi64>{bufferization.buffer_layout = affine_map<(d0) -> (d0)>, bufferization.writable = true}) -> i64
  attributes {cinm.available_platforms = [#upmem]} {
  %4 = cinm.op.reduce add (%x) : tensor<67108864xi64> -> i64
  return %4 : i64
}
