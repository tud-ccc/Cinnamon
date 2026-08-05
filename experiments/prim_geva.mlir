//#! cinm-opt --cinm-assign-platforms --cinm-isolate-compute-blocks --upmem-infer-accelerator --split-input-file

// -----
#upmem = #upmem.platform<type = v1A, dimensions = 32x64x24>

func.func @geva_4MB(%x: tensor<1048576xi32>, %y: tensor<1048576xi32>, %c: i32, %d: i32) -> tensor<1048576xi32>
  attributes {cinm.available_platforms = [#upmem]} {
  %3 = cinm.compute -> tensor<1048576xi32> attributes {cinm.available_platforms = [#upmem]} {
    %cs = tensor.splat %c : tensor<1048576xi32>
    %ds = tensor.splat %d : tensor<1048576xi32>
    %1 = cinm.op.elementwise mul %x, %cs : tensor<1048576xi32>
    %2 = cinm.op.elementwise mul %y, %ds : tensor<1048576xi32>
    %3 = cinm.op.elementwise add %1, %2 : tensor<1048576xi32>
    cinm.yield %3 : tensor<1048576xi32>
  }
  return %3 : tensor<1048576xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dimensions = 32x64x24>

func.func @geva_64MB(%x: tensor<16777216xi32>, %y: tensor<16777216xi32>, %c: i32, %d: i32) -> tensor<16777216xi32>
  attributes {cinm.available_platforms = [#upmem]} {
  %3 = cinm.compute -> tensor<16777216xi32> attributes {cinm.available_platforms = [#upmem]} {
    %cs = tensor.splat %c : tensor<16777216xi32>
    %ds = tensor.splat %d : tensor<16777216xi32>
    %1 = cinm.op.elementwise mul %x, %cs : tensor<16777216xi32>
    %2 = cinm.op.elementwise mul %y, %ds : tensor<16777216xi32>
    %3 = cinm.op.elementwise add %1, %2 : tensor<16777216xi32>
    cinm.yield %3 : tensor<16777216xi32>
  }
  return %3 : tensor<16777216xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dimensions = 32x64x24>

func.func @geva_256MB(%x: tensor<67108864xi32>, %y: tensor<67108864xi32>, %c: i32, %d: i32) -> tensor<67108864xi32>
  attributes {cinm.available_platforms = [#upmem]} {
  %3 = cinm.compute -> tensor<67108864xi32> attributes {cinm.available_platforms = [#upmem]} {
    %cs = tensor.splat %c : tensor<67108864xi32>
    %ds = tensor.splat %d : tensor<67108864xi32>
    %1 = cinm.op.elementwise mul %x, %cs : tensor<67108864xi32>
    %2 = cinm.op.elementwise mul %y, %ds : tensor<67108864xi32>
    %3 = cinm.op.elementwise add %1, %2 : tensor<67108864xi32>
    cinm.yield %3 : tensor<67108864xi32>
  }
  return %3 : tensor<67108864xi32>
}
