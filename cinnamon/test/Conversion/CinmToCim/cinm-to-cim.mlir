// RUN: cinm-opt --convert-cinm-to-cim %s | cinm-opt | FileCheck %s

// CHECK-LABEL: simple
func.func @simple(%t0: tensor<6x6xi32>, %t1: tensor<6x6xi32>, %t2 : tensor<6xi32>) {

// CHECK %0 = cim.acquire -> !cim.deviceId
    %result = cinm.compute -> tensor<6xi32> {
// CHECK %1 = cim.op.gemm %arg0, %arg1 : tensor<6x6xi32>, tensor<6x6xi32> -> !cim.future<6x6xi32>
            %r0 = cinm.op.gemm %t0, %t1 : (tensor<6x6xi32>, tensor<6x6xi32>) -> tensor<6x6xi32>
// CHECK %2 = cim.op.gemv %1, %arg2 : !cim.future<6x6xi32>, tensor<6xi32> -> !cim.future<6xi32>
            %r1 = cinm.op.gemv %r0, %t2 : (tensor<6x6xi32>, tensor<6xi32>) -> tensor<6xi32>
// CHECK %3 = cim.barrier %2 : !cim.future<6xi32> -> tensor<6xi32>
            cinm.yield %r1: tensor<6xi32>
    }
// CHECK cim.release %0 : !cim.deviceId

    return
}

