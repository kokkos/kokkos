#!/bin/bash

echo "SOURCE this script!!"

#export TRILINOS_DIR=${HOME}/trilinos/Trilinos
export TRILINOS_DIR=${PWD}/../..

# Load modules
module purge
source ${TRILINOS_DIR}/cmake/std/atdm/load-env.sh cuda-9.2-rdc-opt
#module swap cmake/3.6.2 cmake/3.12.3
#source $TRILINOS_DIR/cmake/std/atdm/load-env.sh cuda-9.2-debug-Kepler37

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
  -DKokkos_ENABLE_Cuda_Relocatable_Device_Code=ON \
  -DKOKKOS_ENABLE_DEPRECATED_CODE=OFF \
  -DKokkos_SOURCE_DIR_OVERRIDE:STRING=kokkos \
  -DKokkosKernels_SOURCE_DIR_OVERRIDE:STRING=kokkos-kernels \
$TRILINOS_DIR

# -DTrilinos_ENABLE_TESTS=ON -DTrilinos_ENABLE_${PACKAGE1}=ON \
#  -DKOKKOS_ENABLE_RELOCATABLE_DEVICE_CODE=ON \

# Notes: Compile using ninja
# make NP=32

# Allocate node:
# bsub -J TestKokkos-DepCodeOn -W 07:00 -Is -n 16 -q rhel7W bash

# Run tests
# ctest -j16

# Submit tests as job
# bsub -x -Is -q rhel7W -n 16 ctest -j16
