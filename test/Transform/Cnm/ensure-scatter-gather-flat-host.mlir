// RUN: cinm-opt %s --cnm-ensure-scatter-gather-contiguous='pack-fragmented=true static-only=false' \
// RUN: | FileCheck %s

// A flat host buffer whose leaves each take one contiguous run needs no
// repack, whatever the tiling: element i already sits where the workgroup
// wants it.
//
// Lining the blocks up with a dimension boundary splits the host index, and
// the split of `dpu * 512 + tasklet * 64 + i * 16 + j` at 4 leaves
// `(dpu * 32 + tasklet * 4 + i) mod 4` on the middle dimension. That is `i`,
// since i < 4 and the other terms are multiples of 4 -- but only once the
// simplifier drops the terms the modulus divides. Left standing, the map
// does not name the dimension it indexes, the widest block cannot be derived
// (cnm::deflateScatterMap matches syntactically), and every leaf looks
// fragmented enough to be worth a 4MiB copy.

#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

// Each buffer dimension ends up named by the dimension that indexes it, which
// is what lets the block derivation absorb both and reach one block per leaf.
// CHECK: #[[MAP:.*]] = affine_map<(d0, d1, d2, d3) -> (d0 * 8 + d1, d2, d3)>

// CHECK-LABEL: @flat_host_is_already_contiguous
func.func @flat_host_is_already_contiguous(%host: memref<1048576xi32>) {
  %wg = cnm.workgroup : !cnm.workgroup<#upmem.array<2048x8, <type = v1A, dpus = 2048, tasklets = 24>>>
  %buf = cnm.declare_buffer() for %wg : !cnm.buffer<4x16xi32 on #upmem.array<2048x8, <type = v1A, dpus = 2048, tasklets = 24>>, #upmem.mram>

  // The host is reshaped to line up with the blocks -- a view, not a copy.
  // CHECK:      memref.expand_shape
  // CHECK:      memref.expand_shape
  // CHECK-NOT:  cnm.compact_buffer
  // CHECK:      cnm.scatter %{{.*}}[#[[MAP]]]
  // CHECK-NOT:  cnm.compact_buffer
  cnm.scatter %host into %buf[affine_map<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 64 + d2 * 16 + d3)>] of %wg
    : memref<1048576xi32> into !cnm.buffer<4x16xi32 on #upmem.array<2048x8, <type = v1A, dpus = 2048, tasklets = 24>>, #upmem.mram>
  return
}
