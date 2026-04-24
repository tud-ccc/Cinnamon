// RUN: cinm-opt %s --cinm-isolate-compute-blocks | FileCheck %s


// CHECK-LABEL: simple
// CHECK: %{{.*}} = cinm.compute () -> tensor<2xi32>
// CHECK: %[[c0:.*]] = arith.constant 0
// CHECK-NEXT: tensor.generate
// CHECK: tensor.yield %[[c0]]
func.func @simple(%t00: tensor<6x6xi32>, %t10 : tensor<6xf32> , %m00: memref<6xf32>) -> tensor<2xi32> {
    // cinm.accelerator #cinm.host_platform

    %c0 = arith.constant 0 : i32
    %x = cinm.compute_ -> tensor<2xi32> {
      %x2 = tensor.generate {
        ^bb0(%i : index):
          tensor.yield %c0 : i32
      } : tensor<2xi32>
      cinm.yield %x2 : tensor<2xi32>
    }

    return %x : tensor<2xi32>
}
