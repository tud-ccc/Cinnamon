## Getting Started

### Prerequisites

The easiest way to get everything the build needs is [pixi](https://pixi.sh),
a single binary that installs without root. It puts the whole toolchain into
`.pixi/` in the repository, at the versions pinned in `pixi.lock`: the C++
compiler (Clang 22 with conda-forge's libstdc++), CMake, Ninja, `just`,
ccache, mold, clang-format, pre-commit, Conan, the Vulkan headers and loader,
and Python 3.12 with PyTorch and MLIR's Python dependencies. The compiler
builds against glibc 2.28, so the same toolchain works on any distribution
with glibc 2.28 or newer.

Without pixi, you need:

- A C++20 host compiler: GCC 12 or newer, or Clang 16 or newer
- CMake (at least version 3.28)
- [`just`](https://github.com/casey/just?tab=readme-ov-file#installation)
- Python 3.10–3.12
- For `-enable-gpu` builds, the Vulkan headers

```sh
sudo apt-get install clang ninja-build mold python3.12-dev ccache libvulkan-dev
```

Running GPU code needs a Vulkan driver for your GPU, which comes from the
system either way.

Everything else — LLVM/MLIR, Torch-MLIR, the cost model, and (without pixi)
the Python environment — is set up by the build scripts.

LLVM, Torch-MLIR, the Conan packages and Cinnamon must all be built with one
compiler and one standard library (libstdc++). With pixi, that is pixi's
Clang. Without pixi, the build picks a host compiler itself and exports it to
every sub-build; set `CC` and `CXX` to override the choice.


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
of your own. LLVM is usually downloaded prebuilt (see below), and if you
already have an LLVM build you reuse across projects, you can set
`LLVM_BUILD_DIR`: either way `third-party/llvm` is never cloned. So there is
no need for `--recursive` when cloning:

```sh
git clone https://github.com/tud-ccc/Cinnamon.git
```

The UPMEM SDK is the exception: it is no longer publicly downloadable, so it
cannot be a submodule. Unpack it into `third-party/upmem`, point `UPMEM_HOME`
at it, or build without it using `-no-upmem`.

### Prebuilt LLVM

Building LLVM takes hours, so the build downloads it instead. The CI of
[tud-ccc/cinnamon-llvm](https://github.com/tud-ccc/cinnamon-llvm/actions/workflows/cinnamon-prebuilt.yml)
builds every commit pushed to its `cinnamon` branch, and publishes it as a
release tagged `cinnamon-<first 12 digits of the commit>`. `build-llvm.sh`
downloads the one for the revision `third-party/llvm` is pinned to, unpacks it
into `third-party/llvm-prebuilt` (a 440 MB download), and replaces it when the
pin moves. If there is none, it builds the submodule from source instead.

- It is built for Linux on x86-64 with pixi's toolchain, whose C++ runtime it
  bundles, so it runs on any system with glibc 2.28 or newer.
- Its MLIR Python bindings only work with the Python version in `pixi.toml`
  (3.12), which has to match `cinnamon/pixi.toml` in the fork.
- It is a release build with assertions and line-table debug info, so crash
  backtraces show file and line numbers in LLVM too. The build links LLVM's
  `llvm-symbolizer` into `build/bin`, where LLVM looks for it.

`LLVM_PREBUILT=never` always builds LLVM from source; `LLVM_PREBUILT=always`
fails rather than do that. Setting `LLVM_SOURCE_DIR` or `LLVM_BUILD_DIR`
bypasses the download.

The fork configures LLVM for Cinnamon in `cinnamon/llvm-config.cmake`. The
prebuilt LLVM and source builds of the fork both use that file, so that is
where to change an LLVM option. To move to a new LLVM:

1. Push the commit to the `cinnamon` branch of the fork. That starts its
   "Cinnamon prebuilt LLVM" workflow.
2. Point the submodule at the commit and commit that. Without a checkout of
   the submodule:

   ```sh
   git update-index --cacheinfo 160000,<commit>,third-party/llvm
   ```

Our CI waits for the fork's build of the pinned commit, and fails if that
build failed or there is none. To build a commit that is not on the
`cinnamon` branch:

```sh
gh workflow run cinnamon-prebuilt.yml -R tud-ccc/cinnamon-llvm --ref cinnamon -f llvm-ref=<commit>
```

### Configuration

Configuration is read from a `.env` file in the repository root, and from the
environment. The build scripts load `.env` last, so its values take
precedence.

```sh
CMAKE_GENERATOR=Ninja

# Only for builds with the host compiler (no pixi, or `pixi run -e host`), and
# only if the compiler the build picks is not the one you want.
CC=/usr/bin/gcc-13
CXX=/usr/bin/g++-13

# Building LLVM uses a lot of memory, so it is worth limiting the number of
# parallel compile, link and tablegen jobs. These values suit 32 GiB of RAM.
LLVM_CMAKE_OPTIONS='-DLLVM_CCACHE_BUILD=ON -DLLVM_PARALLEL_COMPILE_JOBS=16 -DLLVM_PARALLEL_LINK_JOBS=2 -DLLVM_PARALLEL_TABLEGEN_JOBS=8'

CINNAMON_CMAKE_OPTIONS='-DBUILD_SHARED_LIBS=ON -DCMAKE_LINKER_TYPE=MOLD'
```

Do not put `-DCMAKE_C_COMPILER` / `-DCMAKE_CXX_COMPILER` in the
`*_CMAKE_OPTIONS` variables; use `CC` and `CXX` so that Conan gets the same
compiler as CMake. The build refuses a compiler override it cannot use.

Each dependency is described by three independent settings: where its sources
are, where its build is, and whether we build it. Setting either path variable
leaves the matching submodule uninitialized.

| Variable | Effect |
|---|---|
| `LLVM_SOURCE_DIR` | Build LLVM from your checkout instead of the submodule |
| `LLVM_BUILD_DIR` | Use an LLVM you have already built; nothing is cloned or built |
| `LLVM_PREBUILT` | `auto` (default), `always` or `never` download a [prebuilt LLVM](#prebuilt-llvm) |
| `LLVM_PREBUILT_URL` | Where to download it from, in place of the fork's releases |
| `TORCH_MLIR_SOURCE_DIR` | Build Torch-MLIR from your checkout instead of the submodule |
| `TORCH_MLIR_INSTALL_DIR` | Use a Torch-MLIR you have already installed |
| `UPMEM_HOME` | Location of the UPMEM SDK |
| `CINNAMON_BUILD_DIR` | Where to build Cinnamon itself (default `build/`) |

### Build

With pixi:

```sh
pixi run configure
```

Without pixi:

```sh
just configure
```

This downloads or builds LLVM, then builds Torch-MLIR and Cinnamon (without
pixi, it first creates the Python venv). It is only needed for the first build; afterwards
`just build` does an incremental build of Cinnamon alone.

`pixi run configure` registers the in-tree Conan recipes, then runs
`just configure -no-python-venv` inside the pixi environment, which takes the
place of the venv. With pixi, use it instead of `just configure`. Every other
recipe works unchanged: enter the environment with `pixi shell` and use `just`
as usual, or prefix a single command, as in `pixi run just test`.

`configure` accepts flags to skip parts of the build:
`-no-torch-mlir`, `-no-upmem`, `-no-llvm`, `-no-python-venv`,
`-no-cinnamon-wheel`, `-enable-gpu`, `-enable-cuda`, `-enable-roc`, plus
`-reconfigure` to force a fresh CMake configure and `-verbose` to show all
output. `pixi run configure` passes them on. Building without the torch
frontend is considerably quicker:

```sh
just configure -no-torch-mlir        # or: pixi run configure -no-torch-mlir
```

`pixi.toml` pins the CPU build of PyTorch; `-enable-cuda` and `-enable-roc`
do not change that.

Each step under `.github/workflows/` is also a standalone script, so you can
redo a single part of the build — for example `.github/workflows/build-llvm.sh`
after moving the LLVM submodule.

#### Using the host compiler

The default pixi environment builds everything, LLVM included, with pixi's
Clang. LLVM and everything linked against it must come from the same compiler
and glibc baseline, so an LLVM built with your system compiler cannot be used
there. To use one anyway, for instance an LLVM you share with other projects,
switch to the `host` environment: the same tools and Python, but no compiler
of its own.

1. Pin the host compiler in `.env`, so that LLVM, Torch-MLIR, the Conan
   packages and Cinnamon are all built with it:

   ```sh
   CC=/usr/bin/clang
   CXX=/usr/bin/clang++
   ```

   For an LLVM you have already built, use the compiler it was built with.
   Its build directory records it:

   ```sh
   grep -E '^CMAKE_(C|CXX)_COMPILER:' /path/to/llvm/build/CMakeCache.txt
   ```

2. If you do not have such an LLVM yet, build one from your own checkout. It
   ends up in `/path/to/llvm-project/build`:

   ```sh
   LLVM_SOURCE_DIR=/path/to/llvm-project pixi run -e host .github/workflows/build-llvm.sh
   ```

   An LLVM built some other way works too, as long as it has MLIR's Python
   bindings, RTTI and exceptions enabled; `cinnamon/llvm-config.cmake` in the
   fork lists the options we use.

3. Point the build at it in `.env`, then use the `host` environment for every
   command:

   ```sh
   LLVM_BUILD_DIR=/path/to/llvm/build
   ```

   ```sh
   pixi run -e host configure
   pixi shell -e host        # then `just build`, `just test` as usual
   ```

Only set `CC` and `CXX` in `.env` for the `host` environment: `.env` takes
precedence over pixi's settings, so they would also replace pixi's compiler
in the default environment.

#### Changing tool or Python versions

Versions are pinned in `pixi.lock`. To change one, edit `pixi.toml`, run
`pixi lock`, and commit both files. All Python packages come from PyPI, not
conda-forge; the comment at the top of `pixi.toml` explains why.