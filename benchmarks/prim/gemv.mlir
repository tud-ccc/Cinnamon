//#! cinm-opt --cinm-assign-platforms --cinm-isolate-compute-blocks --upmem-infer-accelerator --split-input-file

// (A * x) * c as one linalg op: the scaling of x is folded into the
// contraction and recomputed for every row, which is the form ATiM's
// schedule of this benchmark takes and what its transcribed points map onto.
// Written out as linalg rather than as cinm ops because elementwise fusion no
// longer produces it (fuse-with-recompute=false); gemv_norecompute.mlir keeps
// the two-op form for the other side of that comparison.

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>
#A = affine_map<(m, k) -> (m, k)>
#x = affine_map<(m, k) -> (k)>
#c = affine_map<(m, k) -> ()>
#y = affine_map<(m, k) -> (m)>

// %A is the weight operand: same data on every inference, so its
// transfer and any repack of it amortize over the serving lifetime
// (cinm.static, see cinm::isStaticValue). %x is the per-inference input.
func.func @gemv_4MB(%A: tensor<1024x1024xi32> {cinm.static}, %x: tensor<1024xi32>, %c: i32 {cinm.static}) -> tensor<1024xi32>
  attributes {cinm.available_platforms = [#upmem]} {
  %zero = arith.constant 0 : i32
  %5 = cinm.compute -> tensor<1024xi32> {
    %e = tensor.empty() : tensor<1024xi32>
    %init = linalg.fill ins(%zero : i32) outs(%e : tensor<1024xi32>) -> tensor<1024xi32>
    %y = linalg.generic {indexing_maps = [#A, #x, #c, #y], iterator_types = ["parallel", "reduction"]}
        ins(%A, %x, %c : tensor<1024x1024xi32>, tensor<1024xi32>, i32) outs(%init : tensor<1024xi32>) {
    ^bb0(%a: i32, %xk: i32, %ck: i32, %acc: i32):
      %s = arith.muli %xk, %ck : i32
      %p = arith.muli %a, %s : i32
      %r = arith.addi %acc, %p : i32
      linalg.yield %r : i32
    } -> tensor<1024xi32>
    cinm.yield %y : tensor<1024xi32>
  }
  return %5 : tensor<1024xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>
#A = affine_map<(m, k) -> (m, k)>
#x = affine_map<(m, k) -> (k)>
#c = affine_map<(m, k) -> ()>
#y = affine_map<(m, k) -> (m)>

func.func @gemv_64MB(%A: tensor<4096x4096xi32> {cinm.static}, %x: tensor<4096xi32>, %c: i32 {cinm.static}) -> tensor<4096xi32>
  attributes {cinm.available_platforms = [#upmem]} {
  %zero = arith.constant 0 : i32
  %5 = cinm.compute -> tensor<4096xi32> {
    %e = tensor.empty() : tensor<4096xi32>
    %init = linalg.fill ins(%zero : i32) outs(%e : tensor<4096xi32>) -> tensor<4096xi32>
    %y = linalg.generic {indexing_maps = [#A, #x, #c, #y], iterator_types = ["parallel", "reduction"]}
        ins(%A, %x, %c : tensor<4096x4096xi32>, tensor<4096xi32>, i32) outs(%init : tensor<4096xi32>) {
    ^bb0(%a: i32, %xk: i32, %ck: i32, %acc: i32):
      %s = arith.muli %xk, %ck : i32
      %p = arith.muli %a, %s : i32
      %r = arith.addi %acc, %p : i32
      linalg.yield %r : i32
    } -> tensor<4096xi32>
    cinm.yield %y : tensor<4096xi32>
  }
  return %5 : tensor<4096xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>
#A = affine_map<(m, k) -> (m, k)>
#x = affine_map<(m, k) -> (k)>
#c = affine_map<(m, k) -> ()>
#y = affine_map<(m, k) -> (m)>

func.func @gemv_256MB(%A: tensor<8192x8192xi32> {cinm.static}, %x: tensor<8192xi32>, %c: i32 {cinm.static}) -> tensor<8192xi32>
  attributes {cinm.available_platforms = [#upmem]} {
  %zero = arith.constant 0 : i32
  %5 = cinm.compute -> tensor<8192xi32> {
    %e = tensor.empty() : tensor<8192xi32>
    %init = linalg.fill ins(%zero : i32) outs(%e : tensor<8192xi32>) -> tensor<8192xi32>
    %y = linalg.generic {indexing_maps = [#A, #x, #c, #y], iterator_types = ["parallel", "reduction"]}
        ins(%A, %x, %c : tensor<8192x8192xi32>, tensor<8192xi32>, i32) outs(%init : tensor<8192xi32>) {
    ^bb0(%a: i32, %xk: i32, %ck: i32, %acc: i32):
      %s = arith.muli %xk, %ck : i32
      %p = arith.muli %a, %s : i32
      %r = arith.addi %acc, %p : i32
      linalg.yield %r : i32
    } -> tensor<8192xi32>
    cinm.yield %y : tensor<8192xi32>
  }
  return %5 : tensor<8192xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>
#A = affine_map<(m, k) -> (m, k)>
#x = affine_map<(m, k) -> (k)>
#c = affine_map<(m, k) -> ()>
#y = affine_map<(m, k) -> (m)>

func.func @gemv_512MB(%A: tensor<8192x16384xi32> {cinm.static}, %x: tensor<16384xi32>, %c: i32 {cinm.static}) -> tensor<8192xi32>
  attributes {cinm.available_platforms = [#upmem]} {
  %zero = arith.constant 0 : i32
  %5 = cinm.compute -> tensor<8192xi32> {
    %e = tensor.empty() : tensor<8192xi32>
    %init = linalg.fill ins(%zero : i32) outs(%e : tensor<8192xi32>) -> tensor<8192xi32>
    %y = linalg.generic {indexing_maps = [#A, #x, #c, #y], iterator_types = ["parallel", "reduction"]}
        ins(%A, %x, %c : tensor<8192x16384xi32>, tensor<16384xi32>, i32) outs(%init : tensor<8192xi32>) {
    ^bb0(%a: i32, %xk: i32, %ck: i32, %acc: i32):
      %s = arith.muli %xk, %ck : i32
      %p = arith.muli %a, %s : i32
      %r = arith.addi %acc, %p : i32
      linalg.yield %r : i32
    } -> tensor<8192xi32>
    cinm.yield %y : tensor<8192xi32>
  }
  return %5 : tensor<8192xi32>
}
