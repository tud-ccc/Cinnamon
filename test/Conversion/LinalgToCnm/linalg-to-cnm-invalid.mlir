// RUN: cinm-opt %s --split-input-file --convert-linalg-to-cnm -verify-diagnostics

// The distribution is fully determined by `cnm.tile_sizes`, so everything that
// can go wrong is a property of those numbers against the op and the
// workgroup. Each diagnostic has to name both sides: the M7 failure this
// design replaces reported only "numParallelElts (64) % numWgItems (16384)",
// which said nothing about which parameter was wrong.

#m = affine_map<(m, k) -> (m, k)>
#v = affine_map<(m, k) -> (k)>
#r = affine_map<(m, k) -> (m)>
#pf = #upmem.platform<type = v1A, dpus = 64, tasklets = 24>
#acc = #upmem.array<16x1, #pf>

func.func @wrong_arity(%A: tensor<1024x512xi32>, %x: tensor<512xi32>) -> tensor<1024xi32> {
  %init = tensor.empty() : tensor<1024xi32>
  %r = cinm.compute on accelerator #acc -> tensor<1024xi32> {
    // expected-error @below {{expected 2 block size(s) in 'cnm.tile_sizes' (one per iteration dimension), got 1}}
    %g = linalg.contract indexing_maps = [#m, #v, #r]
      {cnm.tile_sizes = array<i64: 64>}
      ins(%A, %x : tensor<1024x512xi32>, tensor<512xi32>)
      outs(%init : tensor<1024xi32>) -> tensor<1024xi32>
    cinm.yield %g : tensor<1024xi32>
  }
  func.return %r : tensor<1024xi32>
}

// -----

#m = affine_map<(m, k) -> (m, k)>
#v = affine_map<(m, k) -> (k)>
#r = affine_map<(m, k) -> (m)>
#pf = #upmem.platform<type = v1A, dpus = 64, tasklets = 24>
#acc = #upmem.array<16x1, #pf>

func.func @indivisible(%A: tensor<1024x512xi32>, %x: tensor<512xi32>) -> tensor<1024xi32> {
  %init = tensor.empty() : tensor<1024xi32>
  %r = cinm.compute on accelerator #acc -> tensor<1024xi32> {
    // expected-error @below {{block size 48 does not divide the extent 1024 of iteration dimension 0}}
    %g = linalg.contract indexing_maps = [#m, #v, #r]
      {cnm.tile_sizes = array<i64: 48, 512>}
      ins(%A, %x : tensor<1024x512xi32>, tensor<512xi32>)
      outs(%init : tensor<1024xi32>) -> tensor<1024xi32>
    cinm.yield %g : tensor<1024xi32>
  }
  func.return %r : tensor<1024xi32>
}

// -----

#m = affine_map<(m, k) -> (m, k)>
#v = affine_map<(m, k) -> (k)>
#r = affine_map<(m, k) -> (m)>
#pf = #upmem.platform<type = v1A, dpus = 64, tasklets = 24>
#acc = #upmem.array<16x1, #pf>

// The tile counts must fill the workgroup exactly. 1024/128 = 8 tiles for 16
// leaves: half the workgroup would idle, and the scatter map would no longer
// be the bijection the gather relies on.
func.func @too_few_tiles(%A: tensor<1024x512xi32>, %x: tensor<512xi32>) -> tensor<1024xi32> {
  %init = tensor.empty() : tensor<1024xi32>
  %r = cinm.compute on accelerator #acc -> tensor<1024xi32> {
    // expected-error @below {{the block sizes produce 8 tile(s) but the workgroup has 16 leaves}}
    %g = linalg.contract indexing_maps = [#m, #v, #r]
      {cnm.tile_sizes = array<i64: 128, 512>}
      ins(%A, %x : tensor<1024x512xi32>, tensor<512xi32>)
      outs(%init : tensor<1024xi32>) -> tensor<1024xi32>
    cinm.yield %g : tensor<1024xi32>
  }
  func.return %r : tensor<1024xi32>
}

// -----

#m = affine_map<(m, k) -> (m, k)>
#v = affine_map<(m, k) -> (k)>
#r = affine_map<(m, k) -> (m)>
#pf = #upmem.platform<type = v1A, dpus = 64, tasklets = 24>
#acc = #upmem.array<16x1, #pf>

// Splitting a float reduction reassociates the sum, so it needs the opt-in.
// The search would otherwise silently change results.
func.func @float_split_needs_optin(%A: tensor<1024x512xf32>, %x: tensor<512xf32>, %y: tensor<1024xf32>) -> tensor<1024xf32> {
  %r = cinm.compute on accelerator #acc -> tensor<1024xf32> {
    // expected-error @below {{splitting reduction dimension 1 16 ways reassociates a floating-point reduction, which changes the result; pass allow-float-reassociation to permit it}}
    %g = linalg.contract indexing_maps = [#m, #v, #r]
      {cnm.tile_sizes = array<i64: 1024, 32>}
      ins(%A, %x : tensor<1024x512xf32>, tensor<512xf32>)
      outs(%y : tensor<1024xf32>) -> tensor<1024xf32>
    cinm.yield %g : tensor<1024xf32>
  }
  func.return %r : tensor<1024xf32>
}

// -----

#pf = #upmem.platform<type = v1A, dpus = 64, tasklets = 24>
#acc = #upmem.array<16x1, #pf>

// Splitting requires an associative combiner with a neutral element.
// Subtraction has neither, and must not be split into partials.
func.func @non_associative_split(%A: tensor<1024x512xi32>, %o: tensor<1024xi32>) -> tensor<1024xi32> {
  %r = cinm.compute on accelerator #acc -> tensor<1024xi32> {
    // expected-error @below {{could not split reduction dimension 1: its combiner was not recognised as one with a neutral element}}
    %g = linalg.reduce ins(%A : tensor<1024x512xi32>) outs(%o : tensor<1024xi32>)
      dimensions = [1]
      {cnm.tile_sizes = array<i64: 1024, 32>}
      (%in: i32, %acc: i32) {
        %s = arith.subi %in, %acc : i32
        linalg.yield %s : i32
      }
    cinm.yield %g : tensor<1024xi32>
  }
  func.return %r : tensor<1024xi32>
}

// -----

#pf = #upmem.platform<type = v1A, dpus = 64, tasklets = 24>
#acc = #upmem.array<16x1, #pf>

// Dropping a dimension is fine -- that is just a broadcast, and the gemv's
// vector operand relies on it. Repeating one is not: the tile would be a
// diagonal slice, which cnm.buffer cannot describe.
#diag = affine_map<(d0, d1) -> (d0, d0)>
#out = affine_map<(d0, d1) -> (d0, d1)>
func.func @not_a_projected_permutation(%a: tensor<64x64xi32>) -> tensor<64x16xi32> {
  %init = tensor.empty() : tensor<64x16xi32>
  %r = cinm.compute on accelerator #acc -> tensor<64x16xi32> {
    // expected-error @below {{cannot distribute operand #0: indexing map affine_map<(d0, d1) -> (d0, d0)> is not a projected permutation}}
    %g = linalg.generic {
        indexing_maps = [#diag, #out],
        iterator_types = ["parallel", "parallel"],
        cnm.tile_sizes = array<i64: 4, 16>}
      ins(%a : tensor<64x64xi32>) outs(%init : tensor<64x16xi32>) {
      ^bb0(%in: i32, %o: i32):
        linalg.yield %in : i32
    } -> tensor<64x16xi32>
    cinm.yield %g : tensor<64x16xi32>
  }
  func.return %r : tensor<64x16xi32>
}

// -----

#m = affine_map<(m, k) -> (m, k)>
#v = affine_map<(m, k) -> (k)>
#r = affine_map<(m, k) -> (m)>

// No enclosing accelerator means there is no workgroup to distribute onto.
func.func @no_accelerator(%A: tensor<1024x512xi32>, %x: tensor<512xi32>) -> tensor<1024xi32> {
  %init = tensor.empty() : tensor<1024xi32>
  // expected-error @below {{is not inside a compute block with an accelerator}}
  %g = linalg.contract indexing_maps = [#m, #v, #r]
    {cnm.tile_sizes = array<i64: 64, 512>}
    ins(%A, %x : tensor<1024x512xi32>, tensor<512xi32>)
    outs(%init : tensor<1024xi32>) -> tensor<1024xi32>
  func.return %g : tensor<1024xi32>
}

// -----

#m = affine_map<(m, k) -> (m, k)>
#v = affine_map<(m, k) -> (k)>
#r = affine_map<(m, k) -> (m)>
#pf = #upmem.platform<type = v1A, dpus = 64, tasklets = 24>
#acc = #upmem.array<16x1, #pf>

// The two forms of the workgroup dim order (design §G3) say the same thing, so
// an op carrying both has not been given a choice, it has been given two.
func.func @order_stated_twice(%A: tensor<1024x512xi32>, %x: tensor<512xi32>, %y: tensor<1024xi32>) -> tensor<1024xi32> {
  %r = cinm.compute on accelerator #acc -> tensor<1024xi32> {
    // expected-error @below {{carries both 'cnm.workgroup_dim_order' and 'cnm.workgroup_dim_order_index', which are two ways of stating the same thing; keep one}}
    %g = linalg.contract indexing_maps = [#m, #v, #r]
      {cnm.tile_sizes = array<i64: 256, 128>,
       cnm.workgroup_dim_order = array<i64: 1, 0, 2>,
       cnm.workgroup_dim_order_index = 1 : i64}
      ins(%A, %x : tensor<1024x512xi32>, tensor<512xi32>)
      outs(%y : tensor<1024xi32>) -> tensor<1024xi32>
    cinm.yield %g : tensor<1024xi32>
  }
  func.return %r : tensor<1024xi32>
}

// -----

#m = affine_map<(m, k) -> (m, k)>
#v = affine_map<(m, k) -> (k)>
#r = affine_map<(m, k) -> (m)>
#pf = #upmem.platform<type = v1A, dpus = 64, tasklets = 24>
#acc = #upmem.array<16x1, #pf>

// A discardable attribute of the wrong type is silently invisible to
// getAttrOfType, so it is checked for rather than ignored: a permutation
// written as an array of anything else would otherwise fall back to the
// default rule without a word.
func.func @order_wrong_type(%A: tensor<1024x512xi32>, %x: tensor<512xi32>, %y: tensor<1024xi32>) -> tensor<1024xi32> {
  %r = cinm.compute on accelerator #acc -> tensor<1024xi32> {
    // expected-error @below {{'cnm.workgroup_dim_order' must be a dense i64 array giving a permutation of the iteration dimensions}}
    %g = linalg.contract indexing_maps = [#m, #v, #r]
      {cnm.tile_sizes = array<i64: 256, 128>,
       cnm.workgroup_dim_order = [1, 0, 2]}
      ins(%A, %x : tensor<1024x512xi32>, tensor<512xi32>)
      outs(%y : tensor<1024xi32>) -> tensor<1024xi32>
    cinm.yield %g : tensor<1024xi32>
  }
  func.return %r : tensor<1024xi32>
}

// -----

#m = affine_map<(m, k) -> (m, k)>
#v = affine_map<(m, k) -> (k)>
#r = affine_map<(m, k) -> (m)>
#pf = #upmem.platform<type = v1A, dpus = 64, tasklets = 24>
#acc = #upmem.array<16x1, #pf>

// The number of distinct orders depends on how many dimensions the block sizes
// actually spread over the workgroup, so the diagnostic reports both.
func.func @order_index_out_of_range(%A: tensor<1024x512xi32>, %x: tensor<512xi32>, %y: tensor<1024xi32>) -> tensor<1024xi32> {
  %r = cinm.compute on accelerator #acc -> tensor<1024xi32> {
    // expected-error @below {{'cnm.workgroup_dim_order_index' is 5, but this op spreads 2 iteration dimension(s) over the workgroup, so it has 2 distinct order(s)}}
    %g = linalg.contract indexing_maps = [#m, #v, #r]
      {cnm.tile_sizes = array<i64: 256, 128>,
       cnm.workgroup_dim_order_index = 5 : i64}
      ins(%A, %x : tensor<1024x512xi32>, tensor<512xi32>)
      outs(%y : tensor<1024xi32>) -> tensor<1024xi32>
    cinm.yield %g : tensor<1024xi32>
  }
  func.return %r : tensor<1024xi32>
}
