#!/bin/bash

# UPMEM_VER="2024.1.0"
# UPMEM_FNAME="upmem-${UPMEM_VER}-Linux-x86_64.tar.gz"

# mkdir -p third-party
# pushd third-party
# wget "http://sdk-releases.upmem.com/${UPMEM_VER}/debian_10/${UPMEM_FNAME}"
# tar -xf ${UPMEM_FNAME}
# mv ${UPMEM_FNAME%.tar.gz} upmem
# popd

# #update package manager
# sudo apt update 

# #install pip
# sudo apt install python3-pip

# #install pybind11 CMake package
# pip install pybind11