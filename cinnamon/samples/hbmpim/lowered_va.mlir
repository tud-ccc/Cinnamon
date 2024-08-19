module {
//   func.func @va(%arg0: tensor<8192x8192xi32>, %arg1: tensor<8192x8192xi32>, %arg2: tensor<8192x8192xi32>) {
//     %c0_i32 = arith.constant 0 : i32
//     %c4 = arith.constant 4 : index
//     %c8192 = arith.constant 8192 : index
//     %c4096 = arith.constant 4096 : index
//     affine.for %arg3 = 0 to %c4 {
//       affine.for %arg4 = 0 to %c4 {
//         %0 = arith.muli %arg3, %c8192 : index
//         %1 = arith.muli %arg4, %c8192 : index
//         %extracted_slice = tensor.extract_slice %arg0[%0, %1] [1, 8192] [1, 1] : tensor<8192x8192xi32> to tensor<8192xi32>
//         %extracted_slice_0 = tensor.extract_slice %arg1[%0, %1] [1, 8192] [1, 1] : tensor<8192x8192xi32> to tensor<8192xi32>
//         %extracted_slice_1 = tensor.extract_slice %arg2[%0, %1] [1, 8192] [1, 1] : tensor<8192x8192xi32> to tensor<8192xi32>
//         hbmpim.launch_func @hbmpim_kernels::@va 
//         %2 = hbmpim.set_dev_config : !hbmpim.configuration<16x64x1x8>
//         %3 = builtin.unrealized_conversion_cast %2 : !hbmpim.configuration<16x64x1x8> to !cnm.workgroup<16x64x1x8>
//         %cst = arith.constant dense<[1024, 8]> : tensor<2xi64>
//         %reshape = tensor.reshape %extracted_slice(%cst) : (tensor<8192xi32>, tensor<2xi64>) -> tensor<1024x8xi32>
//         %reshape_2 = tensor.reshape %extracted_slice_0(%cst) : (tensor<8192xi32>, tensor<2xi64>) -> tensor<1024x8xi32>
//         %reshape_3 = tensor.reshape %extracted_slice_1(%cst) : (tensor<8192xi32>, tensor<2xi64>) -> tensor<1024x8xi32>
//         %c0 = arith.constant 0 : index
//         %c512 = arith.constant 512 : index
//         %c0_4 = arith.constant 0 : index
//         %c128 = arith.constant 128 : index
//         %c256 = arith.constant 256 : index
//         %4 = builtin.unrealized_conversion_cast %reshape : tensor<1024x8xi32> to memref<1024x8xi32>
//         hbmpim.preload_no_replacement %2, %4, %c0_4, %c0 : !hbmpim.configuration<16x64x1x8>, memref<1024x8xi32>, index, index
//         %5 = builtin.unrealized_conversion_cast %reshape_2 : tensor<1024x8xi32> to memref<1024x8xi32>
//         hbmpim.preload_no_replacement %2, %5, %c0_4, %c0 : !hbmpim.configuration<16x64x1x8>, memref<1024x8xi32>, index, index
//       }
//     }
//     return
//   }
  hbmpim.module @hbmpim_kernels {
    hbmpim.func @va() {
      %0 = hbmpim.set_dev_config : !hbmpim.configuration<16x64x1x8>
      %c0 = arith.constant 0 : index
      %c0_0 = arith.constant 0 : index
      hbmpim.sim_preload_no_replacement %0, %c0, %c0_0 : !hbmpim.configuration<16x64x1x8>, index, index
      %c0_1 = arith.constant 0 : index
      %c0_2 = arith.constant 0 : index
      hbmpim.sim_preload_no_replacement %0, %c0_1, %c0_2 : !hbmpim.configuration<16x64x1x8>, index, index
      %c512 = arith.constant 512 : index
      %c0_3 = arith.constant 0 : index
      %c128 = arith.constant 128 : index
      %c256 = arith.constant 256 : index
      %c0_4 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c8192 = arith.constant 8192 : index
      %1 = arith.divui %c512, %c8192 : index
      %2 = arith.subi %1, %c1 : index
      %c8 = arith.constant 8 : index
      %3 = hbmpim.get_pim_cmds ADD, %2, %c0_4, %c0_4 : index, index, index -> !hbmpim.PIMCMDVec<ADD>
      %4 = hbmpim.get_toggle_cond ALL_BANK : index
      hbmpim.set_control bst_hab_pim_, true, %4, false, false : index
      hbmpim.set_control bst_hab_, false, %4, false, false : index
      hbmpim.park_in
      hbmpim.change_pim_mode SB, HAB
      hbmpim.program_crf %3 : !hbmpim.PIMCMDVec<ADD>
      hbmpim.change_pim_mode HAB, HAB_PIM
      scf.for %arg0 = %c0_4 to %1 step %c1 {
        %5 = arith.muli %arg0, %c8 : index
        scf.for %arg1 = %c0_4 to %c1 step %c2 {
          hbmpim.add_transaction_all false, %c0_4, %arg1, %c0_3, %5, null_bst_, true, %c8 : index, index, index, index, index
          hbmpim.add_transaction_all false, %c0_4, %arg1, %c256, %5, null_bst_, true, %c8 : index, index, index, index, index
          hbmpim.add_transaction_all true, %c0_4, %arg1, %c128, %5, null_bst_, true, %c8 : index, index, index, index, index
        }
      }
      hbmpim.change_pim_mode HAB_PIM, HAB
      hbmpim.change_pim_mode HAB, SB
      hbmpim.return
    }
  }
}