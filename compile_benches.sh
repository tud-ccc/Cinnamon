./cinnamon/build/bin/cinm-opt --cinm-tiling --cse ./cinnamon/testbench/va.mlir > ./cinnamon/testbench/generated/va.tiled.mlir
./cinnamon/build/bin/cinm-opt --cinm-tiling --cse ./cinnamon/testbench/va.mlir 
./cinnamon/build/bin/cinm-opt --convert-cinm-to-cnm --func-bufferize --lower-affine --cse ./cinnamon/testbench/generated/va.tiled.mlir > ./cinnamon/testbench/generated/va.cnm.mlir
./cinnamon/build/bin/cinm-opt --convert-cinm-to-cnm --func-bufferize --lower-affine --cse ./cinnamon/testbench/generated/va.tiled.mlir 
./cinnamon/build/bin/cinm-opt --convert-cnm-to-upmem -convert-linalg-to-affine-loops --lower-affine --upmem-outline-kernel --cse ./cinnamon/testbench/generated/va.cnm.mlir 
./cinnamon/build/bin/cinm-opt --convert-cnm-to-upmem -convert-linalg-to-affine-loops --lower-affine --upmem-outline-kernel --cse ./cinnamon/testbench/generated/va.cnm.mlir > ./cinnamon/testbench/generated/va.upmem.mlir
./cinnamon/build/bin/cinm-translate --mlir-to-upmem-cpp ./cinnamon/testbench/generated/va.upmem.mlir > ./cinnamon/testbench/generated/va_dpu.c
./cinnamon/build/bin/cinm-translate --mlir-to-upmem-cpp ./cinnamon/testbench/generated/va.upmem.mlir 

# ./cinnamon/build/bin/cinm-opt ./cinnamon/testbench/generated/va.upmem.mlir \
#     --canonicalize --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm \
# 	--convert-upmem-to-llvm \
#     --fold-memref-alias-ops --expand-strided-metadata --memref-expand --finalize-memref-to-llvm \
# 	--convert-func-to-llvm=use-bare-ptr-memref-call-conv=true --cse --reconcile-unrealized-casts --cse > ./cinnamon/testbench/generated/va.host.ll

