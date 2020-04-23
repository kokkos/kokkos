#!/bin/bash

echo "SOURCE this script!!"

export TRILINOS_DIR=${PWD}/../..

# Load modules
module purge
source ${TRILINOS_DIR}/cmake/std/atdm/load-env.sh cuda-9.2-rdc-release-debug-pt

rm -rf CMake*

# Configure
cmake \
 -GNinja \
 -DTrilinos_CONFIGURE_OPTIONS_FILE:STRING=cmake/std/atdm/ATDMDevEnv.cmake \
 -DTrilinos_ENABLE_TESTS=ON \
 -DTrilinos_ENABLE_ALL_PACKAGES=ON \
  -DKokkos_SOURCE_DIR_OVERRIDE:STRING=kokkos \
  -DKokkosKernels_SOURCE_DIR_OVERRIDE:STRING=kokkos-kernels \
$TRILINOS_DIR

# Notes: 
# Compile using ninja
# make NP=32

# Allocate node:
# bsub -J TestKokkos-DepCodeOn-rdcpt -W 07:00 -Is -n 16 -q rhel7W bash

# Run tests
# ctest -j8

# Submit tests as job
# bsub -x -Is -q rhel7W -n 16 ctest -j8
