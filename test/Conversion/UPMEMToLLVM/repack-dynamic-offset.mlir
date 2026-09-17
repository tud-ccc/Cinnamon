// RUN: cinm-opt %s --convert-upmem-to-llvm | FileCheck %s

// A repack is described to the runtime as constants -- a size and a stride per
// dimension -- so its strides must be static. Its *base offset* need not be,
// and usually is not: the source is a subview taken at a loop index. The
// descriptor carries that offset, so it is added to the pointer rather than to
// the stride table, and only the map's own constant term is folded in here.

// CHECK-LABEL: func.func @compact_from_dynamic_offset
//       CHECK:   %[[SUB:.*]] = builtin.unrealized_conversion_cast %subview
//   CHECK-DAG:   %[[BASE:.*]] = llvm.extractvalue %[[SUB]][1]
//   CHECK-DAG:   %[[OFF:.*]] = llvm.extractvalue %[[SUB]][2]
//       CHECK:   %[[TERM:.*]] = llvm.mlir.constant(0 : i64)
//       CHECK:   %[[IDX:.*]] = llvm.add %[[OFF]], %[[TERM]]
//       CHECK:   %[[PTR:.*]] = llvm.getelementptr %[[BASE]][%[[IDX]]]
//       CHECK:   llvm.call @upmemrt_compact({{.*}}, %[[PTR]],
func.func @compact_from_dynamic_offset(%big: memref<64x256xi32>,
                                       %dst: memref<8x16xi32>, %i: index) {
  %sub = memref.subview %big[%i, 0] [8, 16] [1, 1]
      : memref<64x256xi32> to memref<8x16xi32, strided<[256, 1], offset: ?>>
  cnm.compact_buffer %sub into %dst [affine_map<(d0, d1) -> (d0, d1)>]
      : memref<8x16xi32, strided<[256, 1], offset: ?>> into memref<8x16xi32>
  return
}

// -----

// The map may also carry a constant term of its own -- here d1 + 4, against a
// unit innermost stride -- which is folded in alongside the dynamic base.

// CHECK-LABEL: func.func @compact_map_constant_term
//       CHECK:   %[[OFF:.*]] = llvm.extractvalue %{{.*}}[2]
//       CHECK:   %[[TERM:.*]] = llvm.mlir.constant(4 : i64)
//       CHECK:   llvm.add %[[OFF]], %[[TERM]]
func.func @compact_map_constant_term(%big: memref<64x256xi32>,
                                     %dst: memref<8x16xi32>, %i: index) {
  %sub = memref.subview %big[%i, 0] [8, 32] [1, 1]
      : memref<64x256xi32> to memref<8x32xi32, strided<[256, 1], offset: ?>>
  cnm.compact_buffer %sub into %dst [affine_map<(d0, d1) -> (d0, d1 + 4)>]
      : memref<8x32xi32, strided<[256, 1], offset: ?>> into memref<8x16xi32>
  return
}

// -----

// The mirror: expand writes through the map, so it is the *target* whose
// offset is dynamic and folded into the pointer.

// CHECK-LABEL: func.func @expand_to_dynamic_offset
//       CHECK:   %[[SUB:.*]] = builtin.unrealized_conversion_cast %subview
//   CHECK-DAG:   %[[BASE:.*]] = llvm.extractvalue %[[SUB]][1]
//   CHECK-DAG:   %[[OFF:.*]] = llvm.extractvalue %[[SUB]][2]
//       CHECK:   %[[IDX:.*]] = llvm.add %[[OFF]], %{{.*}}
//       CHECK:   %[[PTR:.*]] = llvm.getelementptr %[[BASE]][%[[IDX]]]
//       CHECK:   llvm.call @upmemrt_expand(%[[PTR]],
func.func @expand_to_dynamic_offset(%big: memref<64x256xi32>,
                                    %src: memref<8x16xi32>, %i: index) {
  %sub = memref.subview %big[%i, 0] [8, 16] [1, 1]
      : memref<64x256xi32> to memref<8x16xi32, strided<[256, 1], offset: ?>>
  cnm.expand_buffer %src into %sub [affine_map<(d0, d1) -> (d0, d1)>]
      : memref<8x16xi32> into memref<8x16xi32, strided<[256, 1], offset: ?>>
  return
}

// -----

// The *target* may sit at a run-time offset too: a slot of a stacked staging
// buffer, one slot per slice of a static tensor. Its pointer is the stack's
// base plus the slot's offset, not the base alone.

// CHECK-LABEL: func.func @compact_into_slot
//       CHECK:   %[[SLOT:.*]] = builtin.unrealized_conversion_cast %subview
//   CHECK-DAG:   %[[BASE:.*]] = llvm.extractvalue %[[SLOT]][1]
//   CHECK-DAG:   %[[OFF:.*]] = llvm.extractvalue %[[SLOT]][2]
//       CHECK:   %[[PTR:.*]] = llvm.getelementptr %[[BASE]][%[[OFF]]]
//       CHECK:   llvm.call @upmemrt_compact(%[[PTR]],
func.func @compact_into_slot(%src: memref<8x16xi32, strided<[256, 1]>>,
                             %stack: memref<4x8x16xi32>, %l: index) {
  %slot = memref.subview %stack[%l, 0, 0] [1, 8, 16] [1, 1, 1]
      : memref<4x8x16xi32> to memref<8x16xi32, strided<[16, 1], offset: ?>>
  cnm.compact_buffer %src into %slot [affine_map<(d0, d1) -> (d0, d1)>]
      : memref<8x16xi32, strided<[256, 1]>> into memref<8x16xi32, strided<[16, 1], offset: ?>>
  return
}
