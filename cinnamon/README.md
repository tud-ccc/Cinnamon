
# Setup instructions

- Create a file named `.env` in the clone directory
- In this file, write `LLVM_BUILD_DIR="<path/to/llvm-project/build"` with an absolute path, eg
```shell
LLVM_BUILD_DIR="/home/clem/Documents/CCC/cinnamon-oot/llvm/build"
```
It should point to the build directory, whatever it is named.

Your LLVM version should be
```
6f89431c3d4de87df6d76cf7ffa73bfa881607b7
```
which you need to fetch from https://github.com/oowekyala/llvm-project

### Linker config

To reduce link times, install the `mold` linker and
```shell
echo "CMAKE_LINKER_TYPE=MOLD\n" >> .env
```