

The goal of this idea is to allow efficient offloading of the QKV matvecs in the following network:
```mlir

	%x, %kc2, %vc2 = scf.for %layer = %c0 to %c6 step %c1 iter_args(%x = %content_row, %kc0 = %kc, %vc0 = %vc) -> (tensor<768xf32>, tensor<6x1024x768xf32>, tensor<6x1024x768xf32>) {
    ...
		// qkv matmuls
		%wqs = tensor.extract_slice %wq [%layer, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
		%wks = tensor.extract_slice %wk [%layer, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
		%wvs = tensor.extract_slice %wv [%layer, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>

		%kdest = tensor.extract_slice %kc0 [%layer, %pos, 0] [1, 1, 768] [1, 1, 1] :   tensor<6x1024x768xf32> to tensor<768xf32>
		%vdest = tensor.extract_slice %vc0 [%layer, %pos, 0] [1, 1, 768] [1, 1, 1] :   tensor<6x1024x768xf32> to tensor<768xf32>

    %q = cinm.op.gemv %wqs, %xb : tensor<768x768xf32>, tensor<768xf32> -> tensor<768xf32>
    %k = cinm.op.gemv %wks, %xb : tensor<768x768xf32>, tensor<768xf32> -> tensor<768xf32>
    %v = cinm.op.gemv %wvs, %xb : tensor<768x768xf32>, tensor<768xf32> -> tensor<768xf32>
    %kc1 = tensor.insert_slice %k into %kc0[%layer, %pos, 0] [1, 1, 768] [1, 1, 1] :  tensor<768xf32> into tensor<6x1024x768xf32>
  }
```

(There are more gemvs in this loop, overall 7, where the weights depend only on the layer. The other gemvs have shape <768, 2048> or <2048, 768>).

The thing to realize is:
- The gemvs have no dependency between each other and can be scheduled in parallel. Therefore they should share a single budget for the DPU allocation.
- The weights are dynamically selected on each layer iteration. But if you can distribute the full weight tensors on the DPUs, then you can convert host-side weight selection into device-side selection in MRAM.

With N DPUs available, the full optimization objective becomes:
- Find a schedule for gemv(M=768 K=768) which satisfies:
  - The DPU count is D <= N/3
  - It is realizable in one kernel launch (all of the weight tensor 768x768 fits in the MRAM of D dpus)
  - And as a stretch goal, the MRAM of D dpus can fit 6 times the weight tensor. This is what allows eliminating weight transfers throughout the different iterations.

Honestly the pain points here are mostly:
- are the QKV matmuls even the hotspot in this network?


How would the IR look like
```mlir
  cnm.scatter %wq[(r, d, t, l, j, k) -> ()] // scatter full tensor 6xi

	%x, %kc2, %vc2 = scf.for %layer = %c0 to %c6 step %c1 iter_args(%x = %content_row, %kc0 = %kc, %vc0 = %vc) -> (tensor<768xf32>, tensor<6x1024x768xf32>, tensor<6x1024x768xf32>) {
    ...
		// qkv matmuls
		%wqs = tensor.extract_slice %wq [%layer, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
		%wks = tensor.extract_slice %wk [%layer, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
		%wvs = tensor.extract_slice %wv [%layer, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>

		%kdest = tensor.extract_slice %kc0 [%layer, %pos, 0] [1, 1, 768] [1, 1, 1] :   tensor<6x1024x768xf32> to tensor<768xf32>
		%vdest = tensor.extract_slice %vc0 [%layer, %pos, 0] [1, 1, 768] [1, 1, 1] :   tensor<6x1024x768xf32> to tensor<768xf32>

    cnm.scatter %layer // scatter layer count into workgroup
    cnm.launch {
      ^bb0(%wq: memref<6x768x768xf32>, %layer: index):

        %wqs = memref.subview %wq [%layer, 0, 0] [1, 768, 768] [1, 1, 1]

        //

    }
    %q = cinm.op.gemv %wqs, %xb : tensor<768x768xf32>, tensor<768xf32> -> tensor<768xf32>
    %k = cinm.op.gemv %wks, %xb : tensor<768x768xf32>, tensor<768xf32> -> tensor<768xf32>
    %v = cinm.op.gemv %wvs, %xb : tensor<768x768xf32>, tensor<768xf32> -> tensor<768xf32>
    %kc1 = tensor.insert_slice %k into %kc0[%layer, %pos, 0] [1, 1, 768] [1, 1, 1] :  tensor<768xf32> into tensor<6x1024x768xf32>
  }
```
