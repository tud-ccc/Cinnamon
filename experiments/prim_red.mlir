//#! cinm-opt --cinm-assign-platforms --cinm-isolate-compute-blocks --upmem-infer-accelerator --split-input-file

// 4MB 524288  64MB 8388608  256MB 34554432  512MB 67108864

// -----
#upmem = #upmem.platform<type = v1A, dimensions = 32x64x24>

func.func @red_4MB(%x: tensor<524288xi32>) -> i32 
  attributes {cinm.available_platforms = [#upmem]} {
  %4 = cinm.compute on platform #upmem -> i32 {
    %4 = cinm.op.reduce add (%x) {dimension=0} : tensor<524288xi32> -> i32
    cinm.yield %4: i32
  }
  return %4 : i32
}

// -----
#upmem = #upmem.platform<type = v1A, dimensions = 32x64x24>

func.func @red_64MB(%x: tensor<8388608xi32>) -> i32 
  attributes {cinm.available_platforms = [#upmem]} {
  %4 = cinm.compute on platform #upmem -> i32 {
    %4 = cinm.op.reduce add (%x) {dimension=0} : tensor<8388608xi32> -> i32
    cinm.yield %4: i32
  }
  return %4 : i32
}

// -----
#upmem = #upmem.platform<type = v1A, dimensions = 32x64x24>

func.func @red_256MB(%x: tensor<34554432xi32>) -> i32
  attributes {cinm.available_platforms = [#upmem]} {
  %4 = cinm.compute on platform #upmem -> i32 {
    %4 = cinm.op.reduce add (%x) {dimension=0} : tensor<34554432xi32> -> i32
    cinm.yield %4: i32
  }
  return %4 : i32
}

// -----
#upmem = #upmem.platform<type = v1A, dimensions = 32x64x24>

func.func @red_512MB(%x: tensor<67108864xi32>) -> i32
  attributes {cinm.available_platforms = [#upmem]} {
  %4 = cinm.compute on platform #upmem -> i32 {
    %4 = cinm.op.reduce add (%x) {dimension=0} : tensor<67108864xi32> -> i32
    cinm.yield %4: i32
  }
  return %4 : i32
}