#!/bin/bash

echo "SOURCE this script!!"

export TRILINOS_DIR=${PWD}/../..

# Load modules
module purge
source ${TRILINOS_DIR}/cmake/std/atdm/load-env.sh Trilinos-atdm-waterman-cuda-9.2-rdc-release-debug-pt

# Packages
PACKAGE1=Tpetra
PACKAGE2=Sacado
PACKAGE3=Stokhos
PACKAGE4=MueLu
PACKAGE5=Intrepid2
PACKAGE6=Ifpack2
PACKAGE7=Panzer
PACKAGE8=Phalanx
PACKAGE9=Stratimikos
PACKAGE10=Belos


rm -rf CMake*

# Configure
cmake \
 -GNinja \
 -DTrilinos_CONFIGURE_OPTIONS_FILE:STRING=cmake/std/atdm/ATDMDevEnv.cmake \
 -DTrilinos_ENABLE_TESTS=ON \
  -DTrilinos_ENABLE_${PACKAGE1}=ON \
  -DTrilinos_ENABLE_${PACKAGE2}=ON \
  -DTrilinos_ENABLE_${PACKAGE3}=ON \
  -DTrilinos_ENABLE_${PACKAGE4}=ON \
  -DTrilinos_ENABLE_${PACKAGE5}=ON \
  -DTrilinos_ENABLE_${PACKAGE6}=ON \
  -DTrilinos_ENABLE_${PACKAGE7}=ON \
  -DTrilinos_ENABLE_${PACKAGE8}=ON \
  -DTrilinos_ENABLE_${PACKAGE9}=ON \
  -DTrilinos_ENABLE_${PACKAGE10}=ON \
  -DKokkos_SOURCE_DIR_OVERRIDE:STRING=kokkos \
  -DKokkosKernels_SOURCE_DIR_OVERRIDE:STRING=kokkos-kernels \
$TRILINOS_DIR


# Notes: 
# Compile using ninja
# make NP=32

# Allocate node:
# bsub -J TestKokkos-DepCodeOn -W 07:00 -Is -n 16 -q rhel7W bash

# Run tests
# ctest -j8

# Submit tests as job
# bsub -x -Is -q rhel7W -n 16 ctest -j8
