// RUN: cinm-opt %s --split-input-file --canonicalize --mlir-print-local-scope | FileCheck %s
// RUN: cinm-opt %s --split-input-file --canonicalize --mlir-print-op-generic --mlir-print-local-scope | FileCheck %s --check-prefix=STORED
// What is printed is what is stored, so canonicalizing the printed form again
// reproduces it exactly.
// RUN: cinm-opt %s --split-input-file --canonicalize | cinm-opt --split-input-file --canonicalize --mlir-print-local-scope | FileCheck %s

// A scatter/gather map is pointwise: it names a host index for every buffer
// element. Canonicalization only simplifies it over its own domain, which the
// workgroup and buffer shapes give exactly. Nothing here says what travels as
// one block -- see cnm-to-upmem-block-derivation.mlir, where a consumer that
// moves blocks works that out from the map and the host layout.

#wg = #upmem.array<4x2, <type = v1A, dpus = 4096, tasklets = 1>>

// The stored attribute is the one that was written, verbatim.
// CHECK-LABEL: func.func @pointwise_is_a_fixpoint
// CHECK:       cnm.scatter %{{.*}}[affine_map<(d0, d1, d2) -> (d0, d1 * 2, d2)>]
// STORED:      "cnm.scatter"{{.*}}<{scatterMap = affine_map<(d0, d1, d2) -> (d0, d1 * 2, d2)>}> : (tensor<4x3x8xi32>
func.func @pointwise_is_a_fixpoint(%a: tensor<4x3x8xi32>) {
  %wg = cnm.workgroup : !cnm.workgroup<#wg>
  %buf = cnm.declare_buffer() for %wg : !cnm.buffer<8xi32 on #wg>
  cnm.scatter %a into %buf[affine_map<(d0, d1, d2) -> (d0, d1 * 2, d2)>] of %wg
      : tensor<4x3x8xi32> into !cnm.buffer<8xi32 on #wg>
  cnm.free_workgroup %wg : !cnm.workgroup<#wg>
  return
}

// -----

#wg = #upmem.array<4x2, <type = v1A, dpus = 4096, tasklets = 1>>

// The domain is bounded by the workgroup and buffer extents, so a floordiv or
// mod that cannot reach its second value folds away. d0 < 4 and d1 < 2 here.

// CHECK-LABEL: func.func @simplified_over_its_domain
// CHECK:       cnm.gather %{{.*}}[affine_map<(d0, d1, d2) -> (d0, d1, d2)>]
func.func @simplified_over_its_domain(%out: tensor<4x2x8xi32>) {
  %wg = cnm.workgroup : !cnm.workgroup<#wg>
  %buf = cnm.declare_buffer() for %wg : !cnm.buffer<8xi32 on #wg>
  %g = cnm.gather %buf[affine_map<(d0, d1, d2) -> (d0 mod 4, d1 mod 2, d2)>] of %wg
      into %out : !cnm.buffer<8xi32 on #wg> into tensor<4x2x8xi32>
  cnm.free_workgroup %wg : !cnm.workgroup<#wg>
  return
}

// -----

#wg = #upmem.array<4x2, <type = v1A, dpus = 4096, tasklets = 1>>

// A leaf need not cover the host dimension its buffer indexes: here each one
// writes the first two of eight elements, which the map says and the shapes
// could not.

// CHECK-LABEL: func.func @host_dimension_is_longer
// CHECK:       cnm.scatter %{{.*}}[affine_map<(d0, d1, d2) -> (d0, d1, d2)>]
func.func @host_dimension_is_longer(%a: tensor<4x2x8xi32>) {
  %wg = cnm.workgroup : !cnm.workgroup<#wg>
  %buf = cnm.declare_buffer() for %wg : !cnm.buffer<2xi32 on #wg>
  cnm.scatter %a into %buf[affine_map<(d0, d1, d2) -> (d0, d1, d2)>] of %wg
      : tensor<4x2x8xi32> into !cnm.buffer<2xi32 on #wg>
  cnm.free_workgroup %wg : !cnm.workgroup<#wg>
  return
}
