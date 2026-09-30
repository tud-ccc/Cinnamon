// RUN: cinm-translate --mlir-to-upmem-cpp %s | FileCheck %s

// Private WRAM buffers are stack arrays padded to the 8-byte transfer
// granularity, in elements. The stack size in the header is charged for those
// same padded sizes: the kernel reserve (256) + 8 + 8 + 256 + 8 = 536.

// CHECK: UPMEM-TRANSLATE: COMPILE_k:16:536:k;
// CHECK-DAG: __dma_aligned int32_t {{v[0-9]+}}[2];
// CHECK-DAG: __dma_aligned int32_t {{v[0-9]+}}[2];
// CHECK-DAG: __dma_aligned int32_t {{v[0-9]+}}[64];
// CHECK-DAG: __dma_aligned int8_t {{v[0-9]+}}[8];
upmem.dpu_program @k() tasklets(16) {
  %a = memref.alloca() : memref<i32, #upmem.wram>
  %b = memref.alloca() : memref<2x1x1xi32, #upmem.wram>
  %c = memref.alloca() : memref<1x1x64xi32, #upmem.wram>
  %d = memref.alloca() : memref<3xi8, #upmem.wram>
  upmem.return
}
