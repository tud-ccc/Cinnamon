// RUN: cinm-opt %s --split-input-file --cinm-assign-platforms | FileCheck %s --check-prefix=OPEN
// RUN: cinm-opt %s --split-input-file --cinm-assign-platforms=require-profitable=true | FileCheck %s --check-prefix=GATED
// RUN: cinm-opt %s --split-input-file --cinm-assign-platforms=require-profitable=true -verify-diagnostics -o /dev/null

// The gate only ever fires with require-profitable; the default has to keep
// wrapping everything it can run, which is what every existing pipeline and
// every frozen benchmark depends on.

#upmem = #upmem.platform<type = v1A, dpus = 2560, tasklets = 24>

// A gemv whose matrix is resident: scattered once, then never re-sent, while
// the host re-reads it from DRAM on every call. The case the array wins.

// OPEN-LABEL: @gemv_resident_weight
// OPEN: cinm.compute
// GATED-LABEL: @gemv_resident_weight
// GATED: cinm.compute
func.func @gemv_resident_weight(%A: tensor<8192x8192xi32> {cinm.static},
                                %x: tensor<8192xi32>) -> tensor<8192xi32>
  attributes {cinm.available_platforms = [#upmem]} {
  %y = cinm.op.gemv %A, %x : tensor<8192x8192xi32>, tensor<8192xi32> -> tensor<8192xi32>
  return %y : tensor<8192xi32>
}

// -----

#upmem = #upmem.platform<type = v1A, dpus = 2560, tasklets = 24>

// The same gemv with the matrix streamed in. Nothing to amortize, and the
// result is 32 KB, so the verdict rests on scatter against the host's DRAM
// path alone. At the top of the array the calibrated scatter edges DRAM by
// a few percent (16.2 GB/s at 2560 DPUs against 15), so this passes -- by
// that margin, and only because nothing sizeable comes back; see @va.

// OPEN-LABEL: @gemv_streamed_weight
// OPEN: cinm.compute
// GATED-LABEL: @gemv_streamed_weight
// GATED: cinm.compute
func.func @gemv_streamed_weight(%A: tensor<8192x8192xi32>,
                                %x: tensor<8192xi32>) -> tensor<8192xi32>
  attributes {cinm.available_platforms = [#upmem]} {
  %y = cinm.op.gemv %A, %x : tensor<8192x8192xi32>, tensor<8192xi32> -> tensor<8192xi32>
  return %y : tensor<8192xi32>
}

// -----

#upmem = #upmem.platform<type = v1A, dpus = 2560, tasklets = 24>

// The same streamed gemv on a host that declares a faster DRAM path. The
// margin above was a few percent, so 20 GB/s is enough to lose it: the host
// description is read from the function's platform list, and an entry for
// the host there is not a candidate for offloading.

// OPEN-LABEL: @gemv_streamed_weight_fast_host
// OPEN: cinm.compute
// GATED-LABEL: @gemv_streamed_weight_fast_host
// GATED-NOT: cinm.compute
func.func @gemv_streamed_weight_fast_host(%A: tensor<8192x8192xi32>,
                                          %x: tensor<8192xi32>) -> tensor<8192xi32>
  attributes {cinm.available_platforms = [#cinm.host_platform<dram_bytes_per_second = 2.0e10>, #upmem]} {
  // expected-remark @below {{not offloaded: no static operand}}
  %y = cinm.op.gemv %A, %x : tensor<8192x8192xi32>, tensor<8192xi32> -> tensor<8192xi32>
  return %y : tensor<8192xi32>
}

// -----

#upmem = #upmem.platform<type = v1A, dpus = 2560, tasklets = 24>

// Vector add: pure traffic, no reuse, no resident operand, and a result as
// large as each input, so the gather -- the slow direction -- is paid in
// full. This is `va`, and rejecting it is the model agreeing with what PrIM
// measured.

// OPEN-LABEL: @va
// OPEN: cinm.compute
// GATED-LABEL: @va
// GATED-NOT: cinm.compute
func.func @va(%x: tensor<16777216xi32>, %y: tensor<16777216xi32>) -> tensor<16777216xi32>
  attributes {cinm.available_platforms = [#upmem]} {
  // expected-remark @below {{not offloaded: no static operand}}
  %r = cinm.op.elementwise add %x, %y : tensor<16777216xi32>
  return %r : tensor<16777216xi32>
}

// -----

#upmem = #upmem.platform<type = v1A, dpus = 2560, tasklets = 24>

// A cinm.compute written by hand is an instruction, not a suggestion: the
// gate is not consulted, and no remark is emitted. This is the override for
// benchmarks that mean to run an unprofitable op on the device anyway.

// GATED-LABEL: @va_forced
// GATED: cinm.compute
// GATED-SAME: cinm.available_platforms
func.func @va_forced(%x: tensor<16777216xi32>, %y: tensor<16777216xi32>) -> tensor<16777216xi32>
  attributes {cinm.available_platforms = [#upmem]} {
  %r = cinm.compute -> tensor<16777216xi32> {
    %a = cinm.op.elementwise add %x, %y : tensor<16777216xi32>
    cinm.yield %a : tensor<16777216xi32>
  }
  return %r : tensor<16777216xi32>
}

// -----

#upmem = #upmem.platform<type = v1A, dpus = 2560, tasklets = 24>

// Dynamic shapes: the model cannot read the op, which is not evidence
// against it. Unknown must not mean rejected.

// GATED-LABEL: @dynamic_shape
// GATED: cinm.compute
func.func @dynamic_shape(%A: tensor<?x?xi32> {cinm.static},
                         %x: tensor<?xi32>) -> tensor<?xi32>
  attributes {cinm.available_platforms = [#upmem]} {
  %y = cinm.op.gemv %A, %x : tensor<?x?xi32>, tensor<?xi32> -> tensor<?xi32>
  return %y : tensor<?xi32>
}

// -----

#upmem = #upmem.platform<type = v1A, dpus = 2560, tasklets = 24>
#id = affine_map<(d0) -> (d0)>
#scalar = affine_map<(d0) -> ()>

// A scalar operand (a row's maximum handed to the body) is measurable -- it
// is one number -- so the op is read like any other elementwise op and
// rejected for having no static operand, not waved through as unknown.

// GATED-LABEL: @scalar_operand
// GATED-NOT: cinm.compute
func.func @scalar_operand(%x: tensor<1024xi32>, %m: i32) -> tensor<1024xi32>
  attributes {cinm.available_platforms = [#upmem]} {
  %init = tensor.empty() : tensor<1024xi32>
  // expected-remark @below {{not offloaded: no static operand}}
  %y = linalg.generic {indexing_maps = [#id, #scalar, #id], iterator_types = ["parallel"]}
      ins(%x, %m : tensor<1024xi32>, i32) outs(%init : tensor<1024xi32>) {
  ^bb0(%a: i32, %b: i32, %o: i32):
    %d = arith.subi %a, %b : i32
    linalg.yield %d : i32
  } -> tensor<1024xi32>
  return %y : tensor<1024xi32>
}

// -----

#upmem = #upmem.platform<type = v1A, dpus = 2560, tasklets = 24>

// The layer loop: each iteration's weight is one slice of the static stack,
// selected by the loop index. Every slice can stay resident (one slot each),
// so the gate prices the slice's bytes against the host exactly as it would
// an unrolled member's, and the op is offloaded inside the loop.

// GATED-LABEL: @layer_loop
// GATED: scf.for
// GATED: cinm.compute
func.func @layer_loop(%W: tensor<4x8192x8192xi32> {cinm.static},
                      %x0: tensor<8192xi32>) -> tensor<8192xi32>
  attributes {cinm.available_platforms = [#upmem]} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %r = scf.for %l = %c0 to %c4 step %c1 iter_args(%x = %x0) -> (tensor<8192xi32>) {
    %w = tensor.extract_slice %W[%l, 0, 0] [1, 8192, 8192] [1, 1, 1] : tensor<4x8192x8192xi32> to tensor<8192x8192xi32>
    %y = cinm.op.gemv %w, %x : tensor<8192x8192xi32>, tensor<8192xi32> -> tensor<8192xi32>
    scf.yield %y : tensor<8192xi32>
  }
  return %r : tensor<8192xi32>
}
