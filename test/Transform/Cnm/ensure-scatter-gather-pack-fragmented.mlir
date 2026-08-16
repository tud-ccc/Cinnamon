// RUN: cinm-opt %s --cnm-ensure-scatter-gather-contiguous \
// RUN: | FileCheck %s --check-prefix=KEEP
// RUN: cinm-opt %s --cnm-ensure-scatter-gather-contiguous=pack-fragmented=true \
// RUN: | FileCheck %s --check-prefix=PACK
// RUN: cinm-opt %s \
// RUN:   --cnm-ensure-scatter-gather-contiguous='pack-fragmented=true static-only=true' \
// RUN: | FileCheck %s --check-prefix=STATIC

#wg = #upmem.array<2x1, <type = v1A, dpus = 4096, tasklets = 1>>

// The alias block precedes every function, so the maps @divided_dim expects
// are bound here rather than next to it.
// PACK-DAG: #[[DIV_COMPACT:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1 + d2 * 4, d3)>
// PACK-DAG: #[[DIV_SCATTER:.*]] = affine_map<(d0, d1, d2, d3) -> (d0 floordiv 4, d0 mod 4, d2, d3)>

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
// The allocation carries the conclusion too: afterwards it is the only thing
// left saying its contents are the same on every inference, which is what the
// backend asks when deciding how to time the transfer out of it.
// The workgroup's tasklet dimension is absent from the packed shape: this
// map never reads along it, so one copy per DPU is all there is to hold.
// PACK:       %[[P:.*]] = memref.alloc() {cinm.static} : memref<2x2x8xi32>
// PACK-NEXT:  cnm.compact_buffer %arg0 into %[[P]][#[[M:.*]]] {cinm.static} : memref<4x8xi32> into memref<2x2x8xi32>
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

#wg8 = #upmem.array<8x1, <type = v1A, dpus = 4096, tasklets = 1>>

// A workgroup mapping that folds two tile indices into the one DPU index
// divides that index, which the repack cannot walk: it moves its target with
// one constant stride per dimension, and `w floordiv 4` makes the source
// offset jump every fourth step of w rather than advance by a stride. The
// divisor becomes a dimension boundary, and then it does not.
//
// Leaf w reads rows (w mod 4) and (w mod 4) + 4 of plane (w floordiv 4), so
// its share is two runs and the packed DPU dimension splits 8 into 2x4. Both
// results are then linear in the packed indices.

// PACK-LABEL: func.func @divided_dim
// STATIC-LABEL: func.func @divided_dim
// STATIC:     cnm.compact_buffer %arg0 into %{{.*}} {cinm.static}
// PACK:       %[[P:.*]] = memref.alloc() {cinm.static} : memref<2x4x2x8xi32>
// PACK-NEXT:  cnm.compact_buffer %arg0 into %[[P]][#[[DIV_COMPACT]]] {cinm.static} : memref<2x8x8xi32> into memref<2x4x2x8xi32>
// The scatter reaches the split dimensions by taking its leaf index apart.
// PACK-NEXT:  cnm.scatter %[[P]] into %{{.*}}[#[[DIV_SCATTER]]] of
func.func @divided_dim(%a: memref<2x8x8xi32> {cinm.static}) {
  %wg = cnm.workgroup : !cnm.workgroup<#wg8>
  %b = cnm.declare_buffer() for %wg : !cnm.buffer<2x8xi32 on #wg8>
  cnm.scatter %a into %b[affine_map<(w0, w1, b0, b1) -> (w0 floordiv 4, w0 mod 4 + b0 * 4, b1)>] of %wg
      : memref<2x8x8xi32> into !cnm.buffer<2x8xi32 on #wg8>
  cnm.free_workgroup %wg : !cnm.workgroup<#wg8>
  return
}

// A divisor that does not divide the extent it sits in cannot become a
// boundary, so no repack is expressible and the several-blocks transfer is
// left to the backend rather than emitting IR that fails to lower.

// PACK-LABEL:   func.func @indivisible
// PACK-NOT:     cnm.compact_buffer
// STATIC-LABEL: func.func @indivisible
// STATIC-NOT:   cnm.compact_buffer
func.func @indivisible(%a: memref<2x8x8xi32> {cinm.static}) {
  %wg = cnm.workgroup : !cnm.workgroup<#wg8>
  %b = cnm.declare_buffer() for %wg : !cnm.buffer<2x8xi32 on #wg8>
  cnm.scatter %a into %b[affine_map<(w0, w1, b0, b1) -> (w0 floordiv 3, w0 mod 3 + b0 * 4, b1)>] of %wg
      : memref<2x8x8xi32> into !cnm.buffer<2x8xi32 on #wg8>
  cnm.free_workgroup %wg : !cnm.workgroup<#wg8>
  return
}
