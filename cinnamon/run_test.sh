# ./build/bin/cinm-opt samples/hbmpim/va.mlir 
# ./build/bin/cinm-opt samples/hbmpim/cnm.mlir 
# ./build/bin/cinm-opt samples/hbmpim/cnm_va.mlir --convert-cnm-to-hbmpim
# ./build/bin/cinm-opt samples/hbmpim/test.mlir --convert-cinm-to-cnm
# ./build/bin/cinm-opt samples/hbmpim/test2.mlir 
# ./build/bin/cinm-opt samples/hbmpim/va_hbmpim.mlir
# ./build/bin/cinm-opt samples/hbmpim/test3.mlir 
# ./build/bin/cinm-opt samples/hbmpim/hbmpim_va.mlir --hbmpim-outline-kernel --hbmpim-rewrite-device-calls
# ./build/bin/cinm-translate samples/hbmpim/lowered_va.mlir --mlir-to-hbmpim-cpp

./build/bin/cinm-opt samples/hbmpim/cnm_gemv.mlir --convert-cnm-to-hbmpim --reconcile-unrealized-casts
#  --hbmpim-outline-kernel