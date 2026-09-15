// RUN: cinm-opt --cinm-isolate-compute-blocks --one-shot-bufferize --convert-cinm-to-cim %s | FileCheck %s

// cim is memref-only, so the compute block is bufferized first: the gemm-like
// ops then carry their destination as an `out` operand and have no result.
// The block is isolated from above, so its body reads the operands through
// block arguments, and those have to survive the block being dissolved.

// CHECK-LABEL: func.func @gemm
// CHECK-SAME:      (%[[A:.*]]: tensor<6x6xi32>, %[[B:.*]]: tensor<6x6xi32>)
//       CHECK:   %[[BM:.*]] = bufferization.to_buffer %[[B]]
//       CHECK:   %[[AM:.*]] = bufferization.to_buffer %[[A]]
//       CHECK:   %[[DEV:.*]] = cim.acquire_device
//       CHECK:   %[[XB:.*]] = cim.acquire_crossbar %[[DEV]]
//       CHECK:   %[[OUT:.*]] = memref.alloc() : memref<6x6xi32>
//       CHECK:   %[[FUT:.*]] = cim.op.gemm %[[XB]], %[[AM]], %[[BM]] : {{.*}} -> !cim.future<6x6xi32>
//       CHECK:   %[[RES:.*]] = cim.barrier %[[FUT]] : !cim.future<6x6xi32> -> memref<6x6xi32>
//       CHECK:   memref.copy %[[RES]], %[[OUT]]
//       CHECK:   cim.release_crossbar %[[XB]]
//       CHECK:   cim.release_device %[[DEV]]
//       CHECK:   bufferization.to_tensor %[[OUT]]
func.func @gemm(%t0: tensor<6x6xi32>, %t1: tensor<6x6xi32>) -> tensor<6x6xi32> {
  %r = cinm.compute -> tensor<6x6xi32> {
    %g = cinm.op.gemm %t0, %t1 : tensor<6x6xi32>, tensor<6x6xi32> -> tensor<6x6xi32>
    cinm.yield %g : tensor<6x6xi32>
  }
  return %r : tensor<6x6xi32>
}

// CHECK-LABEL: func.func @gemv
// CHECK-SAME:      (%[[A:.*]]: tensor<6x6xi32>, %[[X:.*]]: tensor<6xi32>)
//       CHECK:   %[[XM:.*]] = bufferization.to_buffer %[[X]]
//       CHECK:   %[[AM:.*]] = bufferization.to_buffer %[[A]]
//       CHECK:   %[[XB:.*]] = cim.acquire_crossbar
//       CHECK:   %[[OUT:.*]] = memref.alloc() : memref<6xi32>
//       CHECK:   %[[FUT:.*]] = cim.op.gemv %[[XB]], %[[AM]], %[[XM]] : {{.*}} -> !cim.future<6xi32>
//       CHECK:   %[[RES:.*]] = cim.barrier %[[FUT]] : !cim.future<6xi32> -> memref<6xi32>
//       CHECK:   memref.copy %[[RES]], %[[OUT]]
//       CHECK:   bufferization.to_tensor %[[OUT]]
func.func @gemv(%t0: tensor<6x6xi32>, %t1: tensor<6xi32>) -> tensor<6xi32> {
  %r = cinm.compute -> tensor<6xi32> {
    %g = cinm.op.gemv %t0, %t1 : tensor<6x6xi32>, tensor<6xi32> -> tensor<6xi32>
    cinm.yield %g : tensor<6xi32>
  }
  return %r : tensor<6xi32>
}
