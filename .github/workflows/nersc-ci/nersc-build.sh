#/bin/bash

# Kokkos nightly tests.
# OpenMP backend with gcc, clang compilers
# OpenMPTarget backend with clang compilers to begin with.

# Avoid this option for now.
#set -e
set -x

cat $0
threads=32


kk_branch="develop"
# Build Kokkos unit tests
if [ $CI ]; 
then
    BASE=${CI_PROJECT_DIR}
    LLVM_INSTALL_BASE="/globl/common/software/nersc/pe/gpu/llvm"
    LLVM_VERSION="nightly-new"
  export PATH=${LLVM_INSTALL_BASE}/${LLVM_VERSION}/bin:$PATH
  export LD_LIBRARY_PATH=${LLVM_INSTALL_BASE}/${LLVM_VERSION}/lib/x86_64-unknown-linux-gnu:${LLVM_INSTALL_BASE}/${LLVM_VERSION}/lib:$LD_LIBRARY_PATH
else
    BASE=$(pwd)
    #module load llvm/nightly
    LLVM_INSTALL_BASE="/global/common/software/nersc/pe/gpu/llvm"
    LLVM_VERSION="nightly-new"
  export PATH=${LLVM_INSTALL_BASE}/${LLVM_VERSION}/bin:$PATH
  export LD_LIBRARY_PATH=${LLVM_INSTALL_BASE}/${LLVM_VERSION}/lib/x86_64-unknown-linux-gnu:${LLVM_INSTALL_BASE}/${LLVM_VERSION}/lib:$LD_LIBRARY_PATH
    LLVM_VERSION="nightly"
fi
KOKKOS_SRC="${BASE}/Kokkos/${kk_branch}"

if [ -d ${KOKKOS_SRC} ]; then
   rm -rf ${KOKKOS_SRC}
fi
git clone --depth 1 --branch ${kk_branch} https://github.com/kokkos/kokkos.git ${KOKKOS_SRC}

cd ${KOKKOS_SRC} && git pull

# Copy the CTestConfig into the main Kokkos dir
cp ${BASE}/CTestConfig.cmake .

# Copy 
cp ${BASE}/CMakeUserPresets.json .

export OMP_NUM_THREADS=${threads}
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# The routine receives cmake_options for each build and the name for build dir as its arguments.
test()
{
    cd ${KOKKOS_SRC}
    build="build_$1"

    # Does not work
    cmake --preset=$1
    cd $build
    cmake \
        -D BUILDNAME=NERSC_$1 \
        ${KOKKOS_SRC}

      cmake --build . -- -j${threads}

      # Do this as a ctest and upload results to dashboard
      ctest -D Nightly
}

module load PrgEnv-gnu
test "omp-gcc12.3"
test "cuda-12.2"

module load PrgEnv-llvm/1.0
test "omp-llvm18"
test "omptarget-llvm18"
