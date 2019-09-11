module purge
module load devpack/20171203/openmpi/2.1.2/intel/18.1.163
# Trilinos now requires cmake version >= 3.10.0
module swap cmake/3.9.0 cmake/3.10.2

export OMP_NUM_THREADS=8
export JENKINS_DO_CUDA=OFF
export JENKINS_DO_OPENMP=OFF
export JENKINS_DO_PTHREAD=ON
export JENKINS_DO_SERIAL=OFF
export JENKINS_DO_COMPLEX=OFF

export JENKINS_ARCH=SKX
export JENKINS_ARCH_CXX_FLAG="-xCORE-AVX512 -mkl"
export JENKINS_ARCH_C_FLAG="-xCORE-AVX512 -mkl"
export BLAS_LIBRARIES="-mkl;${MKLROOT}/lib/intel64/libmkl_intel_lp64.a;${MKLROOT}/lib/intel64/libmkl_intel_thread.a;${MKLROOT}/lib/intel64/libmkl_core.a"
export LAPACK_LIBRARIES=${BLAS_LIBRARIES}

export JENKINS_DO_TESTS=ON
export JENKINS_DO_EXAMPLES=ON
export JENKINS_DO_SHARED=ON

export QUEUE=blake


module load python

