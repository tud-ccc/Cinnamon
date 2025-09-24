
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
This repository supports the ESWEEK tutorial and adds ALPINE support.
Tutorial notebooks live in `tutorial/notebooks`.

### Prerequisites
- Docker

### Run locally
```bash
./build.sh
./start-notebook.sh
```

### Run remotely
```bash
ssh -p 229X esweek25-cim-XX@ios.inf.uos.de
cd Cinnamon
./start-notebook.sh --no-browser --ip=127.0.0.1 --port=8888
```

Forward the notebook port:
```bash
ssh -p 229X -N -L 8888:127.0.0.1:8888 esweek25-cim-XX@ios.inf.uos.de
```



<!-- CONTRIBUTING -->
## Contributing

If you have a suggestion, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
If you want to contribute in any way , that is also **greatly appreciated**.

<!-- LICENSE -->
## License

Distributed under the BSD 2-clause License. See `LICENSE.txt` for more information.

<!-- CONTACT -->
## Contributors

- Hamid Farzaneh (amid.farzaneh@tu-dresden.de)
- Clément Fournier (clement.fournier@tu-dresden.de)
- George M. Kunze (georg_maximilian.kunze@mailbox.tu-dresden.de)
- Asif Ali Khan (asif_ali.khan@tu-dresden.de)
