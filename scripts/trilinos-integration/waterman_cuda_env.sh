#!/bin/bash -el
ulimit -c 0

module purge

module load git
module load devpack/20180517/openmpi/2.1.2/gcc/7.2.0/cuda/9.2.88
module swap openblas/0.2.20/gcc/7.2.0 netlib/3.8.0/gcc/7.2.0
# Trilinos now requires cmake version >= 3.10.0
module swap cmake/3.6.2 cmake/3.12.3
export OMP_NUM_THREADS=8
export JENKINS_DO_CUDA=ON
export JENKINS_DO_OPENMP=OFF
export JENKINS_DO_PTHREAD=OFF
export JENKINS_DO_SERIAL=ON
export JENKINS_DO_COMPLEX=OFF

export JENKINS_ARCH="Power9,Volta70"
export BLAS_LIBRARIES="${BLAS_ROOT}/lib/libblas.a;gfortran;gomp"
export LAPACK_LIBRARIES="${LAPACK_ROOT}/lib/liblapack.a;gfortran;gomp"

export JENKINS_DO_TESTS=ON
export JENKINS_DO_EXAMPLES=ON

export QUEUE=rhel7F

module load python

export CUDA_LAUNCH_BLOCKING=1
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1


export KOKKOS_EXTRA_FLAGS="-DKokkos_ENABLE_CUDA_LAMBDA=ON"
scriptdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "DIR=$scriptdir"
NVCC_WRAPPER=`realpath $scriptdir/../../bin/nvcc_wrapper`
export OMPI_CXX=$NVCC_WRAPPER

