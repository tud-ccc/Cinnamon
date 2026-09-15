
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
#### Building

See [BUILDING.md](BUILDING.md).


### Tests

```sh
just test
```

<!-- USAGE EXAMPLES -->
## Usage
Benchmarks at the `cinm` abstraction live under `benchmarks/`, one directory
per suite: `cinm1/` is the CINM 1.0 flow driven by its own makefile, while
`prim/` and `multiop/` are compiled by the experiment pipelines. Compiling
and running a `cinm1` benchmark goes through `just`, and needs the UPMEM SDK
(for the last step at least):

```sh
just genBench gemv   # compile only; output lands in benchmarks/cinm1/gen/gemv/
just bench gemv      # compile and run
```

The generated code and the intermediate IRs for each bench are written to
`benchmarks/cinm1/gen/`. You can also lower a benchmark by hand: each benchmark file
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
