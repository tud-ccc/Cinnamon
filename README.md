
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
This repository is designed for the ESWEEK tutorial 2025 and adds ALPINE support.
Tutorial notebooks live in `tutorial/notebooks`.

### Prerequisites
- Docker

### Build the Cinnamon image
```bash
docker build -t cinnamon-build .
```

### Build Cinnamon (LLVM, Torch-MLIR, Cinnamon)
Run the build inside the container. The source tree is mounted so build artifacts
persist on the host between runs. `build-alpine.sh` is skipped automatically
when running inside Docker.
```bash
docker run --rm -it \
  -u "$(id -u)":"$(id -g)" \
  -v "$(pwd)":/workspace \
  -e HOME=/workspace \
  cinnamon-build \
  ./build.sh
```

### Build ALPINE / gem5
The ALPINE build manages its own Docker image (`alpine-gem5`). Run this directly
on the host — no Docker-in-Docker required.
```bash
.github/workflows/build-alpine.sh
```

### Start the notebook
The notebooks invoke `alpine-gem5` via Docker, so the host Docker socket is
forwarded into the container. `--group-add` gives the container user access to it.
```bash
docker run --rm -it \
  -u "$(id -u)":"$(id -g)" \
  --group-add "$(stat -c '%g' /var/run/docker.sock)" \
  -v "$(pwd)":/workspace \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e HOME=/workspace \
  -e CINNAMON_HOST_PATH="$(pwd)" \
  -p 8888:8888 \
  cinnamon-build \
  ./start-notebook.sh --no-browser --ip=0.0.0.0 --port=8888
```

Open the URL printed in the terminal (replace `0.0.0.0` with `localhost` if needed).

### Run remotely
Note: The XX is the user id, and will be given during the tutorial.
```bash
ssh -p 229X -L 8888:127.0.0.1:8888 esweek25-cim-XX@ios.inf.uos.de
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

- Hamid Farzaneh (hamid.farzaneh@tu-dresden.de)
- Clément Fournier (clement.fournier@tu-dresden.de)
- George M. Kunze (georg_maximilian.kunze@mailbox.tu-dresden.de)
- Asif Ali Khan (asif_ali.khan@tu-dresden.de)
