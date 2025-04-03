
<br />
<div align="center">
  
  <h3 align="center">CINM (Cinnamon): A Compilation Infrastructure for Heterogeneous Compute In-Memory and Compute Near-Memory Paradigms</h3>

  <p align="center">
    An MLIR Based Compiler Framework for Emerging Architectures
    <br />
    <a href="https://arxiv.org/abs/2301.07486"><strong>Paper Link»</strong></a>
    <br />
  </p>
</div>

<!-- ABOUT THE PROJECT -->
## About The Project

Emerging compute-near-memory (CNM) and compute-in-memory (CIM) architectures have gained considerable attention in recent years, with some now commercially available. However, their programmability remains a significant challenge. These devices typically require very low-level code, directly using device-specific APIs, which restricts their usage to device experts. With Cinnamon, we are taking a step closer to bridging the substantial abstraction gap in application representation between what these architectures expect and what users typically write. The framework is based on MLIR, providing domain-specific and device-specific hierarchical abstractions. This repository includes the sources for these abstractions and the necessary transformations and conversion passes to progressively lower them. It emphasizes conversions to illustrate various intermediate representations (IRs) and transformations to demonstrate certain optimizations.


<!-- GETTING STARTED -->
## Getting Started

This is an example of how you can build the framework locally.

### Prerequisites

CINM depends on a patched version of `LLVM 19.1.3`. This is built automatically.
Additionally, a number of software packages are required to build it:
- CMake (at least version 3.22)
- [`just`](https://github.com/casey/just?tab=readme-ov-file#installation)
- A somewhat recent Python installation (>=3.7?)

On some systems you might need to update your C++ compiler or update the default, e.g. on Ubuntu 24.04:
```sh
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 70 --slave /usr/bin/g++ g++ /usr/bin/g++-13
# Or use another compiler or gcc/g++ version supporting the C++ 20 standard.
```

### Download and Build 

The repository contains a `justfile` that installs all needed dependencies and builds the sources.

* Make sure you install build dependencies:
```sh
sudo apt-get install clang ninja-build mold libvulkan-dev ccache
```
* Clone the repo
  ```sh
  git clone https://github.com/tud-ccc/Cinnamon.git
  ```
* Set up the environment variables in a `.env`-file (in the root)
  ```sh
  # Recommended:
  CMAKE_GENERATOR=Ninja
  CMAKE_C_COMPILER=clang
  CMAKE_CXX_COMPILER=clang++
  CMAKE_LINKER_TYPE=MOLD
  LLVM_CMAKE_OPTIONS='-DLLVM_CCACHE_BUILD=ON'
  TORCH_MLIR_CMAKE_OPTIONS='-DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang'                                                               CINNAMON_CMAKE_OPTIONS='-DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DLLVM_ENABLE_LIBCXX=ON'  
  # You could add your own LLVM dir; the build script won't try to clone and build LLVM
  LLVM_BUILD_DIR=/home/username/projects/Cinnamon/llvm/build/
  ```
* Download, configure, and build dependencies and the sources (without the torch-mlir frontend).
  ```sh
  just configure -no-torch-mlir
  ```

<!-- USAGE EXAMPLES -->
## Usage
All benchmarks at the `cinm` abstraction are in this repository under `cinnamon/benchmarks/`. The `compile-benches.sh` script compiles all the benchmarks using the Cinnamon flow. The generated code and the intermediate IRs for each bench can be found under`cinnamon/benchmarks/generated/`.

   ```sh
   chmod +x compile-benches.sh
   ./compile-benches.sh
   ```
The user can also try running individual benchmarks by manually trying individual conversions. The benchmark files have a comment at the top giving the command used to lower them to the upmem IR.

<!-- ROADMAP -->
## Roadmap

- [x] `cinm`, `cnm` and `cim` abstractions and their necessary conversions 
- [x] The `upmem` abstraction, its conversions and connection to the target
- [x] The `tiling` transformation
- [ ] `PyTorch` Front-end
- [ ] The `xbar` abstraction, conversions and transformations
    - [ ] Associated conversions and transformations
    - [ ] Establishing the backend connection

See the [open issues](https://github.com/tud-ccc/Cinnamon/issues) for a full list of proposed features (and known issues).

<!-- CONTRIBUTING -->
## Contributing

If you have a suggestion, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
If you want to contribute in any way , that is also **greatly appreciated**.

<!-- LICENSE -->
## License

Distributed under the BSD 2-clause License. See `LICENSE.txt` for more information.

<!-- CONTACT -->
## Contributors

- Clément Fournier (clement.fournier@tu-dresden.de)
- Hamid Farzaneh (amid.farzaneh@tu-dresden.de)
- George M. Kunze (georg_maximilian.kunze@mailbox.tu-dresden.de)
- Karl F. A. Friebel (karl.friebel@tu-dresden.de)
- Asif Ali Khan (asif_ali.khan@tu-dresden.de)
