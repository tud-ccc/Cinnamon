#!/bin/bash

UPMEM_VER="2024.1.0"
UPMEM_FNAME="upmem-${UPMEM_VER}-Linux-x86_64.tar.gz"

pushd third-party
wget "http://sdk-releases.upmem.com/${UPMEM_VER}/debian_10/${UPMEM_FNAME}"
tar -xf ${UPMEM_FNAME}
mv ${UPMEM_FNAME%.tar.gz} upmem
popd