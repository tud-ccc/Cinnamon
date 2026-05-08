// RUN: cinm-opt %s --canonicalize | FileCheck %s
// RUN: cinm-opt %s --cinm-isolate-compute-blocks --canonicalize | FileCheck %s


// CHECK-LABEL: simple
func.func @simple(%t00: tensor<6x6xi32>, %t10 : tensor<6xf32> , %m00: memref<6xf32>) -> tensor<6xf32> {

    // CHECK-NOT: cinm.yield %arg
    %d = cinm.compute -> tensor<6xf32> {
      cinm.yield %t10 : tensor<6xf32>
    }

    // CHECK: %[[R:.*]] = arith.addf %arg{{.}}, %arg{{.}} : tensor<6xf32>
    // CHECK-NEXT: cinm.yield %[[R]]
    %d2 = cinm.compute -> tensor<6xf32> {
      %x = arith.addf %d, %d : tensor<6xf32>
      cinm.yield %x : tensor<6xf32>
    }

    return %d2 : tensor<6xf32>
}
