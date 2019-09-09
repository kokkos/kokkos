#!/bin/bash -el
ulimit -c 0

module purge

module load devpack/20180521/openmpi/2.1.2/gcc/7.2.0/cuda/9.2.88
module load netlib/3.8.0/gcc/7.2.0
# Trilinos now requires cmake version >= 3.10.0
module load cmake/3.12.3
export OMP_NUM_THREADS=8
export JENKINS_DO_CUDA=ON
export JENKINS_DO_OPENMP=OFF
export JENKINS_DO_PTHREAD=OFF
export JENKINS_DO_SERIAL=ON
export JENKINS_DO_COMPLEX=OFF

export JENKINS_ARCH="Power8,Kepler37"
export JENKINS_ARCH_CXX_FLAG="-mcpu=power8 -arch=sm_37"
export JENKINS_ARCH_C_FLAG="-mcpu=power8"
export BLAS_LIBRARIES="${BLAS_ROOT}/lib/libblas.a;gfortran;gomp"
export LAPACK_LIBRARIES="${LAPACK_ROOT}/lib/liblapack.a;gfortran;gomp"

export JENKINS_DO_TESTS=ON
export JENKINS_DO_EXAMPLES=ON

export QUEUE=rhel7F

module load python

export CUDA_LAUNCH_BLOCKING=1
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1

export OMPI_CXX=${KOKKOS_PATH}/bin/nvcc_wrapper
