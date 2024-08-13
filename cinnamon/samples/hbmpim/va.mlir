module {
  hbmpim.module @dpu_kernels{
    hbmpim.func @run_va(){
      %input0_row = arith.constant 0 : index
      %input1_row = arith.constant 128 : index
      %result_row = arith.constant 256 : index
      %cst0 = arith.constant 0: index
      %cst1 = arith.constant 1: index
      %cst2 = arith.constant 2: index
      %dim = arith.constant 65536 : index 
      %num_banks = arith.constant 16 : index
      %num_pim_channels = arith.constant 64 : index 
      %num_pim_ranks = arith.constant 1 : index 
      %num_grf = arith.constant 8 : index
      %cst00 = arith.constant 0 : index
      %ranks_pe = arith.muli %num_pim_ranks, %num_grf : index
      %channels_pe = arith.muli %num_pim_channels, %ranks_pe : index 
      %banks_pe = arith.muli %num_banks, %channels_pe : index 
      %num_tile = arith.divui %dim, %banks_pe : index 
      %num_jump_to_be_taken = arith.subi %num_tile, %cst1 : index

      %cmds = hbmpim.get_pim_cmds ADD, %num_jump_to_be_taken, %cst0, %cst0 : index, index, index -> !hbmpim.PIMCMDVec<ADD>
      %toggle_cond = hbmpim.get_toggle_cond ALL_BANK : index
      hbmpim.set_control bst_hab_pim_, true, %toggle_cond, false, false : index
      hbmpim.set_control bst_hab_, false, %toggle_cond, false, false : index
      hbmpim.park_in
      hbmpim.change_pim_mode SB, HAB
      hbmpim.program_crf %cmds : !hbmpim.PIMCMDVec<ADD> 
      hbmpim.change_pim_mode HAB, HAB_PIM
      scf.for %i0 = %cst0 to %num_tile step %cst1 {
        %c = arith.muli %num_grf, %i0: index
        scf.for %i1 = %cst0 to %cst2 step %cst1 {
              hbmpim.add_transaction_all false, %cst0, %i1, %input0_row, %c, null_bst_, true,
                                %num_grf : index, index, index, index, index
              hbmpim.add_transaction_all false, %cst0, %i1, %input1_row, %c, null_bst_, true,
                                %num_grf : index, index, index, index, index
              hbmpim.add_transaction_all true, %cst0, %i1, %result_row, %c, null_bst_, true,
                                %num_grf : index, index, index, index, index
        }
      }
      hbmpim.change_pim_mode HAB_PIM, HAB 
      hbmpim.change_pim_mode HAB, SB 
      hbmpim.return
    }
  }
}