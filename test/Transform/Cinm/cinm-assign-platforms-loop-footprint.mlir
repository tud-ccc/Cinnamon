// RUN: cinm-opt %s --split-input-file "--cinm-assign-platforms=dump-decisions=%t.jsonl" -o /dev/null
// RUN: FileCheck %s < %t.jsonl

// A loop in a linalg body is counted once per trip. Its bounds may reach the
// body as 0-d tensor operands, as they do once the whole-program flow has
// hoisted the constants; those are read back to the constants they hold.
// Integer square root by Newton: 128 lanes x (1 add + 24 trips x 3 ops + the
// 2 constants the body declares) = 9600. The three index tensors are 8 bytes
// each, and static.

// CHECK: "func":"isqrt",{{.*}}"static_bytes":24,{{.*}}"unknown":false,"work_ops":9600}

// A trip count only known at run time leaves the footprint unknown.

// CHECK-NEXT: "func":"isqrt_dynamic",{{.*}}"unknown":true,

#upmem = #upmem.platform<type = v1A, dpus = 2560, tasklets = 24>
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>

func.func @isqrt(%x: tensor<128xi32>) -> tensor<128xi32>
  attributes {cinm.available_platforms = [#upmem]} {
  %lb = arith.constant dense<0> : tensor<index>
  %ub = arith.constant dense<24> : tensor<index>
  %st = arith.constant dense<1> : tensor<index>
  %init = tensor.empty() : tensor<128xi32>
  %r = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map], iterator_types = ["parallel"]}
      ins(%x, %lb, %ub, %st : tensor<128xi32>, tensor<index>, tensor<index>, tensor<index>)
      outs(%init : tensor<128xi32>) {
  ^bb0(%n: i32, %l: index, %u: index, %s: index, %o: i32):
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %n1 = arith.addi %n, %c1 : i32
    %y = scf.for %k = %l to %u step %s iter_args(%y0 = %n1) -> (i32) {
      %q = arith.divsi %n1, %y0 : i32
      %sm = arith.addi %y0, %q : i32
      %h = arith.divsi %sm, %c2 : i32
      scf.yield %h : i32
    }
    linalg.yield %y : i32
  } -> tensor<128xi32>
  return %r : tensor<128xi32>
}

// -----

#upmem = #upmem.platform<type = v1A, dpus = 2560, tasklets = 24>
#map = affine_map<(d0) -> (d0)>

func.func @isqrt_dynamic(%x: tensor<128xi32>, %steps: index) -> tensor<128xi32>
  attributes {cinm.available_platforms = [#upmem]} {
  %init = tensor.empty() : tensor<128xi32>
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%x : tensor<128xi32>) outs(%init : tensor<128xi32>) {
  ^bb0(%n: i32, %o: i32):
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : i32
    %y = scf.for %k = %c0 to %steps step %c1 iter_args(%y0 = %n) -> (i32) {
      %h = arith.divsi %y0, %c2 : i32
      scf.yield %h : i32
    }
    linalg.yield %y : i32
  } -> tensor<128xi32>
  return %r : tensor<128xi32>
}
