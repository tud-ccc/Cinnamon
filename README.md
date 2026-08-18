
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

### Prerequisites

- CMake (at least version 3.28)
- [`just`](https://github.com/casey/just?tab=readme-ov-file#installation)
- Python 3.10–3.12
- A C++20 compiler. The Conan profile in `third-party/conan-profile` assumes
  Clang; adjust it if you build with something else.

```sh
sudo apt-get install clang ninja-build mold libvulkan-dev python3.12-dev ccache
```

Everything else — LLVM/MLIR, Torch-MLIR, the cost model, and the Python
environment — is set up by the build scripts.

### Dependencies and submodules

Source dependencies are git submodules, so the revision this project is known
to work with is recorded in the repository:

| Submodule | Upstream |
|---|---|
| `third-party/llvm` | [tud-ccc/cinnamon-llvm](https://github.com/tud-ccc/cinnamon-llvm), branch `cinnamon` — a fork of LLVM carrying patches we depend on |
| `third-party/torch-mlir` | [llvm/torch-mlir](https://github.com/llvm/torch-mlir), built out of tree against our LLVM |
| `third-party/cnm-cost-model` | [tud-ccc/cnm-cost-model](https://github.com/tud-ccc/cnm-cost-model), built as part of this project |

**You do not have to clone them.** Each build step checks out only the
submodule it is about to build, and only if you have not pointed it at a tree
of your own. If you already have an LLVM build you reuse across projects, set
`LLVM_BUILD_DIR` and `third-party/llvm` is never cloned. So there is no need
for `--recursive` when cloning:

```sh
git clone https://github.com/tud-ccc/Cinnamon.git
```

The UPMEM SDK is the exception: it is no longer publicly downloadable, so it
cannot be a submodule. Unpack it into `third-party/upmem`, point `UPMEM_HOME`
at it, or build without it using `-no-upmem`.

### Configuration

Configuration is read from a `.env` file in the repository root (and from the
environment, which takes precedence).

```sh
# Recommended:
CMAKE_GENERATOR=Ninja
CMAKE_C_COMPILER=clang
CMAKE_CXX_COMPILER=clang++
CMAKE_LINKER_TYPE=MOLD

# Building LLVM uses a lot of memory, so it is worth limiting the number of
# parallel compile, link and tablegen jobs. These values suit 32 GiB of RAM.
LLVM_CMAKE_OPTIONS='-DLLVM_CCACHE_BUILD=ON -DLLVM_PARALLEL_COMPILE_JOBS=16 -DLLVM_PARALLEL_LINK_JOBS=2 -DLLVM_PARALLEL_TABLEGEN_JOBS=8'

TORCH_MLIR_CMAKE_OPTIONS='-DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang'
CINNAMON_CMAKE_OPTIONS='-DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DLLVM_ENABLE_LIBCXX=ON'
```

Each dependency is described by three independent settings: where its sources
are, where its build is, and whether we build it. Setting either path variable
leaves the matching submodule uninitialized.

| Variable | Effect |
|---|---|
| `LLVM_SOURCE_DIR` | Build LLVM from your checkout instead of the submodule |
| `LLVM_BUILD_DIR` | Use an LLVM you have already built; nothing is cloned or built |
| `TORCH_MLIR_SOURCE_DIR` | Build Torch-MLIR from your checkout instead of the submodule |
| `TORCH_MLIR_INSTALL_DIR` | Use a Torch-MLIR you have already installed |
| `UPMEM_HOME` | Location of the UPMEM SDK |
| `CINNAMON_BUILD_DIR` | Where to build Cinnamon itself (default `build/`) |

### Build

```sh
just configure
```

This creates the Python venv, then builds LLVM, Torch-MLIR and Cinnamon in
order. It is only needed for the first build; afterwards `just build` does an
incremental build of Cinnamon alone.

`configure` accepts flags to skip parts of the build:
`-no-torch-mlir`, `-no-upmem`, `-no-llvm`, `-no-python-venv`,
`-no-cinnamon-wheel`, `-enable-gpu`, `-enable-cuda`, `-enable-roc`, plus
`-reconfigure` to force a fresh CMake configure and `-verbose` to show all
output. Building without the torch frontend is considerably quicker:

```sh
just configure -no-torch-mlir
```

Each step under `.github/workflows/` is also a standalone script, so you can
redo a single part of the build — for example `.github/workflows/build-llvm.sh`
after moving the LLVM submodule.

### Tests

```sh
just test
```

<!-- USAGE EXAMPLES -->
## Usage
All benchmarks at the `cinm` abstraction are in this repository under
`testbench/`. Compiling and running one goes through `just`, and needs the
UPMEM SDK (for the last step at least):

```sh
just genBench gemv   # compile only; output lands in testbench/gen/gemv/
just bench gemv      # compile and run
```

The generated code and the intermediate IRs for each bench are written to
`testbench/gen/`. You can also lower a benchmark by hand: each benchmark file
has a comment at the top giving the command that lowers it to the UPMEM IR.
`just cinm-opt` runs the compiler from the build tree without putting it on
your `PATH`.

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
