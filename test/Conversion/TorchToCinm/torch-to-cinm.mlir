// RUN: cinm-opt --split-input-file --convert-torch-to-cinm %s | cinm-opt | FileCheck %s

// CHECK-LABEL: torch.aten.matmul
func.func @torch.aten.matmul(%arg0: !torch.vtensor<[8,16],f32>, %arg1: !torch.vtensor<[16,8],f32>) -> !torch.vtensor<[8,8],f32> {
// CHECK: %[[a:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[8,16],f32> -> tensor<8x16xf32>
// CHECK: %[[b:.*]] = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[16,8],f32> -> tensor<16x8xf32>
// CHECK: %[[r:.*]] = cinm.compute (%[[a1:.*]] = %[[a]] : {{.*}}, %[[b1:.*]] = %[[b]] : {{.*}}) -> tensor<8x8xf32>
// CHECK: %[[r0:.*]] = cinm.op.gemm %[[a1]], %[[b1]] : tensor<8x16xf32>, tensor<16x8xf32> -> tensor<8x8xf32>
// CHECK: cinm.yield %[[r0]] : tensor<8x8xf32>
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[8,16],f32>, !torch.vtensor<[16,8],f32> -> !torch.vtensor<[8,8],f32>
// CHECK: %{{.*}} = torch_c.from_builtin_tensor %[[r]] : tensor<8x8xf32> -> !torch.vtensor<[8,8],f32>
  return %0 : !torch.vtensor<[8,8],f32>
}

// -----


// CHECK-LABEL: torch.aten.mm
func.func @torch.aten.mm(%arg0: !torch.vtensor<[8,16],f32>, %arg1: !torch.vtensor<[16,8],f32>) -> !torch.vtensor<[8,8],f32> {
// CHECK: %[[a:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[8,16],f32> -> tensor<8x16xf32>
// CHECK: %[[b:.*]] = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[16,8],f32> -> tensor<16x8xf32>
// CHECK: %[[r:.*]] = cinm.compute (%[[a1:.*]] = %[[a]] : {{.*}}, %[[b1:.*]] = %[[b]] : {{.*}}) -> tensor<8x8xf32>
// CHECK: %[[r0:.*]] = cinm.op.gemm %[[a1]], %[[b1]] : tensor<8x16xf32>, tensor<16x8xf32> -> tensor<8x8xf32>
// CHECK: cinm.yield %[[r0]] : tensor<8x8xf32>
  %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[8,16],f32>, !torch.vtensor<[16,8],f32> -> !torch.vtensor<[8,8],f32>
// CHECK: %{{.*}} = torch_c.from_builtin_tensor %[[r]] : tensor<8x8xf32> -> !torch.vtensor<[8,8],f32>
  return %0 : !torch.vtensor<[8,8],f32>
}

// -----

// CHECK-LABEL: torch.aten.mv
func.func @torch.aten.mv(%arg0: !torch.vtensor<[8,16],f32>, %arg1: !torch.vtensor<[16],f32>) -> !torch.vtensor<[8],f32> {
// CHECK: %[[a:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[8,16],f32> -> tensor<8x16xf32>
// CHECK: %[[b:.*]] = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[16],f32> -> tensor<16xf32>
// CHECK: %[[r:.*]] = cinm.compute (%[[a1:.*]] = %[[a]] : {{.*}}, %[[b1:.*]] = %[[b]] : {{.*}}) -> tensor<8xf32>
// CHECK: %[[r0:.*]] = cinm.op.gemv %[[a1]], %[[b1]] : tensor<8x16xf32>, tensor<16xf32> -> tensor<8xf32>
// CHECK: cinm.yield %[[r0]] : tensor<8xf32>
  %0 = torch.aten.mv %arg0, %arg1 : !torch.vtensor<[8,16],f32>, !torch.vtensor<[16],f32> -> !torch.vtensor<[8],f32>
// CHECK: %{{.*}} = torch_c.from_builtin_tensor %[[r]] : tensor<8xf32> -> !torch.vtensor<[8],f32>
  return %0 : !torch.vtensor<[8],f32>
}