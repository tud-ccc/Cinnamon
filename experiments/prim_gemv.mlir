//#! cinm-opt --cinm-assign-platforms --cinm-isolate-compute-blocks --upmem-infer-accelerator --split-input-file

// -----
#upmem = #upmem.platform<type = v1A, dimensions = 32x64x24>

func.func @gemv_4MB(%A: tensor<1024x1024xf32>, %x: tensor<1024xf32>) -> tensor<1024xf32> 
  attributes {cinm.available_platforms = [#upmem]} {
  %4 = cinm.op.gemv %A, %x : tensor<1024x1024xf32>, tensor<1024xf32> -> tensor<1024xf32>
  return %4 : tensor<1024xf32>
}

// // -----
// #upmem = #upmem.platform<type = v1A, dimensions = 32x64x24>

// func.func @gemv_64MB(%A: tensor<4096x4096xf32>, %x: tensor<4096xf32>) -> tensor<4096xf32> 
//   attributes {cinm.available_platforms = [#upmem]} {
//   %4 = cinm.op.gemv %A, %x : tensor<4096x4096xf32>, tensor<4096xf32> -> tensor<4096xf32>
//   return %4 : tensor<4096xf32>
// }

// // -----
// #upmem = #upmem.platform<type = v1A, dimensions = 32x64x24>

// func.func @gemv_256MB(%A: tensor<8192x8192xf32>, %x: tensor<8192xf32>) -> tensor<8192xf32> 
//   attributes {cinm.available_platforms = [#upmem]} {
//   %4 = cinm.op.gemv %A, %x : tensor<8192x8192xf32>, tensor<8192xf32> -> tensor<8192xf32>
//   return %4 : tensor<8192xf32>
// }

// // -----
// #upmem = #upmem.platform<type = v1A, dimensions = 32x64x24>

// func.func @gemv_512MB(%A: tensor<8192x16384xf32>, %x: tensor<16384xf32>) -> tensor<8192xf32> 
//   attributes {cinm.available_platforms = [#upmem]} {
//   %4 = cinm.op.gemv %A, %x : tensor<8192x16384xf32>, tensor<16384xf32> -> tensor<8192xf32>
//   return %4 : tensor<8192xf32>
// }