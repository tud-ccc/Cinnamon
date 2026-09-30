// RUN: cinm-opt "--upmem-annotate-costs=costs-csv=%t.csv" %s -o /dev/null
// RUN: FileCheck %s < %t.csv

// The host code around an offloaded block, priced from the host platform in
// scope. Both blocks run the same code: the expand of a gathered 4x1024 i32
// result (16 KiB) into the host's layout, and the loop that sums its four
// partials. The first function names no host, so the bench machine prices
// it; the second declares one next to its accelerator, and the block's own
// accelerator-only list does not hide it.
//
// The expand is a repack at copy bandwidth, and recurs per inference, so it
// is charged. The loop is a roofline: 4096 vectorized i32 adds take
// 4096 * 0.5 ns * 4 / 64 = 128 ns, and 16 KiB of loads plus 4 KiB of stores
// take longer at any plausible stream bandwidth, so the traffic prices it.
//
//   default:  expand 16384 B / 0.63 GB/s = 0.0260063 ms
//             loop   20480 B / 10.8 GB/s = 0.0018963 ms
//   declared: expand 16384 B / 1 GB/s    = 0.016384 ms
//             loop   20480 B / 1 GB/s    = 0.02048 ms

// CHECK: block_id,location,category,label,cost_ms,excluded,block_total_ms
// CHECK:      cpu,"expand",0.0260063,0,
// CHECK-NEXT: cpu,"other",0.0018963,0,
// CHECK:      cpu,"expand",0.016384,0,
// CHECK-NEXT: cpu,"other",0.02048,0,

#upmem = #upmem.platform<type = v1A, dpus = 512, tasklets = 8>
module {
  func.func @bench_machine(%packed: memref<4x1024xi32>, %partials: memref<4x1024xi32>, %out: memref<1024xi32>)
      attributes {cinm.available_platforms = [#upmem]} {
    cinm.compute_block on accelerator #upmem.array<128x4, <type = v1A, dpus = 512, tasklets = 8>> (%p = %packed : memref<4x1024xi32>, %q = %partials : memref<4x1024xi32>, %o = %out : memref<1024xi32>) attributes {cinm.available_platforms = [#upmem]} {
      %c0_i32 = arith.constant 0 : i32
      cnm.expand_buffer %p into %q[affine_map<(d0, d1) -> (d0, d1)>] : memref<4x1024xi32> into memref<4x1024xi32>
      affine.for %i = 0 to 1024 {
        %sum = affine.for %k = 0 to 4 iter_args(%acc = %c0_i32) -> (i32) {
          %v = affine.load %q[%k, %i] : memref<4x1024xi32>
          %s = arith.addi %v, %acc : i32
          affine.yield %s : i32
        }
        affine.store %sum, %o[%i] : memref<1024xi32>
      }
      cinm.yield
    }
    return
  }

  func.func @declared_host(%packed: memref<4x1024xi32>, %partials: memref<4x1024xi32>, %out: memref<1024xi32>)
      attributes {cinm.available_platforms = [#cinm.host_platform<stream_bytes_per_second = 1.0e9, copy_bytes_per_second = 1.0e9>, #upmem]} {
    cinm.compute_block on accelerator #upmem.array<128x4, <type = v1A, dpus = 512, tasklets = 8>> (%p = %packed : memref<4x1024xi32>, %q = %partials : memref<4x1024xi32>, %o = %out : memref<1024xi32>) attributes {cinm.available_platforms = [#upmem]} {
      %c0_i32 = arith.constant 0 : i32
      cnm.expand_buffer %p into %q[affine_map<(d0, d1) -> (d0, d1)>] : memref<4x1024xi32> into memref<4x1024xi32>
      affine.for %i = 0 to 1024 {
        %sum = affine.for %k = 0 to 4 iter_args(%acc = %c0_i32) -> (i32) {
          %v = affine.load %q[%k, %i] : memref<4x1024xi32>
          %s = arith.addi %v, %acc : i32
          affine.yield %s : i32
        }
        affine.store %sum, %o[%i] : memref<1024xi32>
      }
      cinm.yield
    }
    return
  }
}
