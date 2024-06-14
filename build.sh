git clone https://github.com/oowekyala/llvm-project llvm
cd llvm
git checkout cinnamon-llvm
mkdir build   
cd build

cmake -G "Ninja" ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON\
  -DLLVM_BUILD_TOOLS=OFF \
  -DCMAKE_BUILD_TYPE=DEBUG \
  -DBUILD_SHARED_LIBS=ON \
  -DLLVM_TARGETS_TO_BUILD=X86 \
  -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=SPIRV \
  -DLLVM_OPTIMIZED_TABLEGEN=ON

ninja

cd ../..
cd cinnamon 
llvm_prefix=../llvm/build

cmake -S . -B "build" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DLLVM_DIR="$llvm_prefix"/lib/cmake/llvm \
    -DMLIR_DIR="$llvm_prefix"/lib/cmake/mlir \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_LINKER_TYPE=DEFAULT \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON 

cd build && ninja