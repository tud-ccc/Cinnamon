
// -----
// CHECK-LABEL: @gemv
#upmem_platform = #upmem.platform<type=v1A, dimensions = 4x16>
#upmem = #upmem.array<2x4x1, #upmem_platform>

    func.func @gemv(%arg0: tensor<8x1024xi32>, %arg1: tensor<1024xi32>) -> tensor<8xi32> {

// CHECK: %[[cst0:.*]] = arith.constant dense<0> : tensor<8xi32>
// CHECK: %[[wg:.*]] = cnm.workgroup : !cnm.workgroup<{{.*}}>
// CHECK: %[[ba:.*]] = cnm.declare_buffer() for %[[wg]] : !cnm.buffer<1024xi32 on {{.*}}>
// CHECK: cnm.scatter %arg0 into %[[ba]][{{.*}}] of %[[wg]] : tensor<8x1024xi32> into !cnm.buffer<1024xi32 on {{.*}}>
// CHECK: %[[bb:.*]] = cnm.declare_buffer() for %[[wg]] : !cnm.buffer<1024xi32 on {{.*}}>
// CHECK: cnm.scatter %arg1 into %[[bb]][{{.*}}] of %[[wg]] : tensor<1024xi32> into !cnm.buffer<1024xi32 on {{.*}}>
// CHECK: %[[bc:.*]] = cnm.declare_buffer() for %[[wg]] : !cnm.buffer<i32 on {{.*}}>
// CHECK: cnm.scatter %[[cst0]] into %[[bc]][{{.*}}] of %[[wg]] : tensor<8xi32> into !cnm.buffer<i32 on {{.*}}>
// CHECK: cnm.launch %[[wg]] ins(%{{.*}} = %[[ba]] : <1024xi32>, %{{.*}} = %[[bb]] : <1024xi32>) outs(%{{.*}} = %[[bc]] : <i32>) on {{.*}} {
// CHECK:    linalg.contract
// CHECK: %[[emptyres:.*]] = tensor.empty() : tensor<8xi32>
// CHECK: %{{.*}} = cnm.gather %[[bc]][{{.*}}] of %[[wg]] into %[[emptyres]] : !cnm.buffer<i32 on {{.*}}> into tensor<8xi32>
// CHECK: cnm.free_workgroup %[[wg]] : !cnm.workgroup<{{.*}}>
        %r0 = cinm.compute on accelerator #upmem -> tensor<8xi32>  {
            %r = cinm.op.gemv %arg0, %arg1  {cinm.tile_sizes=array<i64: 8, 512>}: tensor<8x1024xi32>, tensor<1024xi32> -> tensor<8xi32>
            cinm.yield %r : tensor<8xi32>
        }
        func.return %r0 : tensor<8xi32>
    }

