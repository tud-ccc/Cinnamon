git clone --depth 1 https://github.com/oowekyala/llvm-project
mkdir llvm-project/build 
cd llvm-project/build

cmake -G "Ninja" ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir;" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON\
  -DCMAKE_BUILD_TYPE=DEBUG \
  -DBUILD_SHARED_LIBS=ON \
  -DLLVM_TARGETS_TO_BUILD=X86 \
  -DLLVM_OPTIMIZED_TABLEGEN=ON

ninja