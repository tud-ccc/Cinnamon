// RUN: cinm-opt %s --cnm-ensure-scatter-gather-contiguous \
// RUN: | FileCheck %s --check-prefix=KEEP
// RUN: cinm-opt %s --cnm-ensure-scatter-gather-contiguous=pack-fragmented=true \
// RUN: | FileCheck %s --check-prefix=PACK
// RUN: cinm-opt %s \
// RUN:   --cnm-ensure-scatter-gather-contiguous='pack-fragmented=true static-only=true' \
// RUN: | FileCheck %s --check-prefix=STATIC

#wg = #upmem.array<2x1, <type = v1A, dpus = 4096, tasklets = 1>>

// A host value that is perfectly contiguous, but whose leaves do not each read
// one run of it: leaf w takes rows w and w+2, so its share is two runs of 8.
// A backend with a several-blocks transfer can move that as it stands, which
// is what the pass does by default.
//
// `pack-fragmented` instead reorders the value into workgroup x buffer order,
// making each leaf's share a single run and the transfer one whole-buffer
// block. That is the only form CINM 1's code generation had, and it is a cost
// trade either way: several transfers per leaf against one repack.

// KEEP-LABEL: func.func @fragmented_static
// KEEP-NOT:   cnm.compact_buffer
// KEEP:       cnm.scatter %arg0 into

// PACK-LABEL: func.func @fragmented_static
// PACK:       %[[P:.*]] = memref.alloc() : memref<2x1x2x8xi32>
// PACK-NEXT:  cnm.compact_buffer %arg0 into %[[P]][#[[M:.*]]] {cinm.static} : memref<4x8xi32> into memref<2x1x2x8xi32>
// PACK-NEXT:  cnm.scatter %[[P]] into %{{.*}}[#{{.*}}] of

// The repack of a declared weight amortizes over the serving lifetime, so
// static-only keeps it.
// STATIC-LABEL: func.func @fragmented_static
// STATIC:       cnm.compact_buffer %arg0 into %{{.*}} {cinm.static}
func.func @fragmented_static(%a: memref<4x8xi32> {cinm.static}) {
  %wg = cnm.workgroup : !cnm.workgroup<#wg>
  %b = cnm.declare_buffer() for %wg : !cnm.buffer<2x8xi32 on #wg>
  cnm.scatter %a into %b[affine_map<(w0, w1, b0, b1) -> (w0 + b0 * 2, b1)>] of %wg
      : memref<4x8xi32> into !cnm.buffer<2x8xi32 on #wg>
  cnm.free_workgroup %wg : !cnm.workgroup<#wg>
  return
}

// The same transfer on a per-inference operand. Its repack would be paid on
// every call, so static-only declines it and leaves the several-blocks
// transfer to the backend.

// PACK-LABEL:   func.func @fragmented_dynamic
// PACK:         cnm.compact_buffer %arg0 into
// PACK-NOT:     cinm.static

// STATIC-LABEL: func.func @fragmented_dynamic
// STATIC-NOT:   cnm.compact_buffer
func.func @fragmented_dynamic(%a: memref<4x8xi32>) {
  %wg = cnm.workgroup : !cnm.workgroup<#wg>
  %b = cnm.declare_buffer() for %wg : !cnm.buffer<2x8xi32 on #wg>
  cnm.scatter %a into %b[affine_map<(w0, w1, b0, b1) -> (w0 + b0 * 2, b1)>] of %wg
      : memref<4x8xi32> into !cnm.buffer<2x8xi32 on #wg>
  cnm.free_workgroup %wg : !cnm.workgroup<#wg>
  return
}
