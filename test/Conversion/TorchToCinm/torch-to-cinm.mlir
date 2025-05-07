// RUN: cinm-opt --convert-torch-to-cinm %s | cinm-opt | FileCheck %s

// CHECK-LABEL: torch.aten.matmul
func.func @torch.aten.matmul(%arg0: !torch.vtensor<[8,16],f32>, %arg1: !torch.vtensor<[16,8],f32>) -> !torch.vtensor<[8,8],f32> {
// CHECK: %0 = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[16,8],f32> -> tensor<16x8xf32>
// CHECK: %1 = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[8,16],f32> -> tensor<8x16xf32>
// CHECK: %2 = cinm.compute -> tensor<8x8xf32>
// CHECK: %4 = cinm.op.gemm %1, %0 : (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
// CHECK: cinm.yield %4 : tensor<8x8xf32>
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[8,16],f32>, !torch.vtensor<[16,8],f32> -> !torch.vtensor<[8,8],f32>
// CHECK: %3 = torch_c.from_builtin_tensor %2 : tensor<8x8xf32> -> !torch.vtensor<[8,8],f32>
  return %0 : !torch.vtensor<[8,8],f32>
}

// CHECK-LABEL: torch.aten.mm
func.func @torch.aten.mm(%arg0: !torch.vtensor<[8,16],f32>, %arg1: !torch.vtensor<[16,8],f32>) -> !torch.vtensor<[8,8],f32> {
// CHECK: %0 = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[16,8],f32> -> tensor<16x8xf32>
// CHECK: %1 = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[8,16],f32> -> tensor<8x16xf32>
// CHECK: %2 = cinm.compute -> tensor<8x8xf32>
// CHECK: %4 = cinm.op.gemm %1, %0 : (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
// CHECK: cinm.yield %4 : tensor<8x8xf32>
  %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[8,16],f32>, !torch.vtensor<[16,8],f32> -> !torch.vtensor<[8,8],f32>
// CHECK: %3 = torch_c.from_builtin_tensor %2 : tensor<8x8xf32> -> !torch.vtensor<[8,8],f32>
  return %0 : !torch.vtensor<[8,8],f32>
}

// CHECK-LABEL: torch.aten.mv
func.func @torch.aten.mv(%arg0: !torch.vtensor<[8,16],f32>, %arg1: !torch.vtensor<[16],f32>) -> !torch.vtensor<[8],f32> {
// CHECK: %0 = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[16],f32> -> tensor<16xf32>
// CHECK: %1 = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[8,16],f32> -> tensor<8x16xf32>
// CHECK: %2 = cinm.compute -> tensor<8xf32>
// CHECK: %4 = cinm.op.gemv %1, %0 : (tensor<8x16xf32>, tensor<16xf32>) -> tensor<8xf32>
// CHECK: cinm.yield %4 : tensor<8xf32>
  %0 = torch.aten.mv %arg0, %arg1 : !torch.vtensor<[8,16],f32>, !torch.vtensor<[16],f32> -> !torch.vtensor<[8],f32>
// CHECK: %3 = torch_c.from_builtin_tensor %2 : tensor<8xf32> -> !torch.vtensor<[8],f32>
  return %0 : !torch.vtensor<[8],f32>
}