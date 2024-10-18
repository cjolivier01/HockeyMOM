#!/bin/bash
TOPDIR="$(pwd)"
set -e
cd external/fast_pytorch_kmeans && python setup.py develop && cd "${TOPDIR}"
cd external/hugin && mkdir -p ./build && cd ./build && cmake .. && sudo make install && cd "${TOPDIR}"
cd openmm && ./build_all.sh && cd "${TOPDIR}"
