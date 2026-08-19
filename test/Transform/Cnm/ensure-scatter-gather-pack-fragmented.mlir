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
// PACK-DAG: #[[UNIT_SCATTER:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, 0, d4)>
// PACK-DAG: #[[REP_COMPACT:.*]] = affine_map<(d0, d1, d2) -> (d0 + d1 * 2, d2)>
// PACK-DAG: #[[REP_SCATTER:.*]] = affine_map<(d0, d1, d2, d3) -> (d0 floordiv 4, d2, d3)>

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
// PACK:       %[[P:.*]] = memref.get_global @{{.*}} : memref<2x2x8xi32> {cinm.static}
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
// PACK:       %[[P:.*]] = memref.get_global @{{.*}} : memref<2x4x2x8xi32> {cinm.static}
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


// A buffer dimension of one element has no host dimension to pair with -- the
// map cannot name it, since a dimension bounded to one value simplifies away.
// The walk pairs it with a unit host dimension split off for the purpose (a
// view), instead of stopping there and declaring the transfer fragmented:
// each leaf's share here is one contiguous run, and packing it would be a
// repack on every inference for nothing. The scatter indexes the unit
// dimension by the constant zero, the degenerate form the block widening
// accepts.

// PACK-LABEL: func.func @unit_buffer_dim
// PACK-NOT:   cnm.compact_buffer
// PACK:       %[[E0:.*]] = memref.expand_shape %arg0 {{\[\[}}0], [1, 2]] output_shape [2, 4, 2]
// PACK-NEXT:  %[[E1:.*]] = memref.expand_shape %[[E0]] {{\[\[}}0], [1, 2], [3]] output_shape [2, 4, 1, 2]
// PACK-NEXT:  cnm.scatter %[[E1]] into %{{.*}}[#[[UNIT_SCATTER]]] of

// KEEP-LABEL: func.func @unit_buffer_dim
// KEEP-NOT:   cnm.compact_buffer

// STATIC-LABEL: func.func @unit_buffer_dim
// STATIC-NOT:   cnm.compact_buffer
func.func @unit_buffer_dim(%a: memref<2x8xi32>) {
  %wg = cnm.workgroup : !cnm.workgroup<#wg>
  %b = cnm.declare_buffer() for %wg : !cnm.buffer<4x1x2xi32 on #wg>
  cnm.scatter %a into %b[affine_map<(w0, w1, b0, b1, b2) -> (w0, b0 * 2 + b2)>] of %wg
      : memref<2x8xi32> into !cnm.buffer<4x1x2xi32 on #wg>
  cnm.free_workgroup %wg : !cnm.workgroup<#wg>
  return
}

// The same shape on the gather side: the destination is written through the
// same views, with no packed intermediate to drain afterwards.

// PACK-LABEL: func.func @unit_buffer_dim_gather
// PACK-NOT:   cnm.compact_buffer
// PACK:       %[[G1:.*]] = memref.expand_shape %{{.*}} output_shape [2, 4, 1, 2]
// PACK-NEXT:  cnm.gather %{{.*}}[#[[UNIT_SCATTER]]] of %{{.*}} into %[[G1]]
// PACK-NOT:   cnm.expand_buffer
func.func @unit_buffer_dim_gather(%out: memref<2x8xi32>) {
  %wg = cnm.workgroup : !cnm.workgroup<#wg>
  %b = cnm.declare_buffer() for %wg : !cnm.buffer<4x1x2xi32 on #wg>
  cnm.gather %b[affine_map<(w0, w1, b0, b1, b2) -> (w0, b0 * 2 + b2)>] of %wg into %out
      : !cnm.buffer<4x1x2xi32 on #wg> into memref<2x8xi32>
  cnm.free_workgroup %wg : !cnm.workgroup<#wg>
  return
}


#wg8r = #upmem.array<8x1, <type = v1A, dpus = 4096, tasklets = 1>>

// A divided DPU dimension whose mod part the map never reads is replication:
// leaves 4q..4q+3 all read plane q, rows interleaved as in @divided_dim. The
// mod part stays out of the packed shape -- one plane's copy per *value* of
// it would be 4x the operand -- and out of the scatter's view of the packed
// buffer, so those leaves' transfers point at the same packed slice instead.


// PACK-LABEL: func.func @replicated_divided_dim
// PACK:       %[[R:.*]] = memref.get_global @{{.*}} : memref<2x2x8xi32> {cinm.static}
// PACK-NEXT:  cnm.compact_buffer %arg0 into %[[R]][#[[REP_COMPACT]]] {cinm.static} : memref<4x8xi32> into memref<2x2x8xi32>
// PACK-NEXT:  cnm.scatter %[[R]] into %{{.*}}[#[[REP_SCATTER]]] of

// The operand is static, so static-only packs it too -- to the same
// replication-free shape.
// STATIC-LABEL: func.func @replicated_divided_dim
// STATIC:      cnm.compact_buffer %arg0 into %{{.*}} {cinm.static} : memref<4x8xi32> into memref<2x2x8xi32>
func.func @replicated_divided_dim(%a: memref<4x8xi32> {cinm.static}) {
  %wg = cnm.workgroup : !cnm.workgroup<#wg8r>
  %b = cnm.declare_buffer() for %wg : !cnm.buffer<2x8xi32 on #wg8r>
  cnm.scatter %a into %b[affine_map<(w0, w1, b0, b1) -> (w0 floordiv 4 + b0 * 2, b1)>] of %wg
      : memref<4x8xi32> into !cnm.buffer<2x8xi32 on #wg8r>
  cnm.free_workgroup %wg : !cnm.workgroup<#wg8r>
  return
}

// The same map on a gather keeps one copy per value of the mod part: leaves
// 4q..4q+3 all *write* plane q, and collapsing them onto one packed slice
// would turn the program's write-order question into concurrent DMA into the
// same bytes. The copies keep those writes apart; what the host then reads
// out of the expand is one of them, as it would have been one of the DMAs.


// PACK-LABEL: func.func @replicated_divided_dim_gather
// PACK:       %[[RG:.*]] = memref.get_global @{{.*}} : memref<2x4x2x8xi32>
// PACK-NEXT:  cnm.gather %{{.*}}[#[[DIV_SCATTER]]] of %{{.*}} into %[[RG]]
// PACK-NEXT:  cnm.expand_buffer %[[RG]] into %arg0
func.func @replicated_divided_dim_gather(%out: memref<4x8xi32>) {
  %wg = cnm.workgroup : !cnm.workgroup<#wg8r>
  %b = cnm.declare_buffer() for %wg : !cnm.buffer<2x8xi32 on #wg8r>
  cnm.gather %b[affine_map<(w0, w1, b0, b1) -> (w0 floordiv 4 + b0 * 2, b1)>] of %wg into %out
      : !cnm.buffer<2x8xi32 on #wg8r> into memref<4x8xi32>
  cnm.free_workgroup %wg : !cnm.workgroup<#wg8r>
  return
}
