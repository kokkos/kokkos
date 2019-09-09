#!/bin/bash

update_kokkos_devices() {
   SEARCH_TEXT="*$1*"
   if [[ $KOKKOS_DEVICES == $SEARCH_TEXT ]]; then
      echo kokkos devices already includes $SEARCH_TEXT
   else
      if [ "$KOKKOS_DEVICES" = "" ]; then
         KOKKOS_DEVICES="$1"
         echo reseting kokkos devices to $KOKKOS_DEVICES
      else
         KOKKOS_DEVICES="${KOKKOS_DEVICES},$1"
         echo appending to kokkos devices $KOKKOS_DEVICES
      fi
   fi
}


KOKKOS_DO_EXAMPLES=ON

while [[ $# > 0 ]]
do
  key="$1"

  case $key in
    --kokkos-path*)
      KOKKOS_PATH="${key#*=}"
      ;;
    --qthreads-path*)
      QTHREADS_PATH="${key#*=}"
      ;;
    --hpx-path*)
      HPX_PATH="${key#*=}"
      ;;
    --prefix*)
      PREFIX="${key#*=}"
      ;;
    --with-cuda)
      update_kokkos_devices Cuda
      CUDA_PATH_NVCC=$(command -v nvcc)
      CUDA_PATH=${CUDA_PATH_NVCC%/bin/nvcc}
      ;;
    # Catch this before '--with-cuda*'
    --with-cuda-options*)
      KOKKOS_CUDA_OPT="${key#*=}"
      ;;
    --with-cuda*)
      update_kokkos_devices Cuda
      CUDA_PATH="${key#*=}"
      ;;
    --with-rocm)
      update_kokkos_devices ROCm
      ;;
    --with-openmp)
      update_kokkos_devices OpenMP
      ;;
    --with-pthread)
      update_kokkos_devices Pthread
      ;;
    --with-serial)
      update_kokkos_devices Serial
      ;;
    --with-qthreads*)
      update_kokkos_devices Qthreads
      if [ -z "$QTHREADS_PATH" ]; then
        QTHREADS_PATH="${key#*=}"
      fi
      ;;
    --with-hpx-options*)
      KOKKOS_HPX_OPT="${key#*=}"
      ;;
    --with-hpx*)
      update_kokkos_devices HPX
      if [ -z "$HPX_PATH" ]; then
        HPX_PATH="${key#*=}"
      fi
      ;;
    --with-devices*)
      DEVICES="${key#*=}"
      PARSE_DEVICES=$(echo $DEVICES | tr "," "\n")
      for DEVICE_ in $PARSE_DEVICES
      do 
         update_kokkos_devices $DEVICE_
      done
      ;;
    --with-gtest*)
      GTEST_PATH="${key#*=}"
      ;;
    --with-hwloc*)
      HWLOC_PATH="${key#*=}"
      ;;
    --with-memkind*)
      MEMKIND_PATH="${key#*=}"
      ;;
    --arch*)
      KOKKOS_ARCH="${key#*=}"
      ;;
    --cxxflags*)
      CXXFLAGS="${key#*=}"
      ;;
    --cxxstandard*)
      KOKKOS_CXX_STANDARD="${key#*=}"
      ;;
    --ldflags*)
      LDFLAGS="${key#*=}"
      ;;
    --debug|-dbg)
      KOKKOS_DEBUG=yes
      ;;
    --make-j*)
      echo "Warning: ${key} is deprecated"
      echo "Call make with appropriate -j flag"
      ;;
    --no-examples)
      KOKKOS_DO_EXAMPLES=OFF
      ;;
    --compiler*)
      COMPILER="${key#*=}"
      CNUM=$(command -v ${COMPILER} 2>&1 >/dev/null | grep "no ${COMPILER}" | wc -l)
      if [ ${CNUM} -gt 0 ]; then
        echo "Invalid compiler by --compiler command: '${COMPILER}'"
        exit
      fi
      if [[ ! -n  ${COMPILER} ]]; then
        echo "Empty compiler specified by --compiler command."
        exit
      fi
      CNUM=$(command -v ${COMPILER} | grep ${COMPILER} | wc -l)
      if [ ${CNUM} -eq 0 ]; then
        echo "Invalid compiler by --compiler command: '${COMPILER}'"
        exit
      fi
      # ... valid compiler, ensure absolute path set 
      WCOMPATH=$(command -v $COMPILER)
      COMPDIR=$(dirname $WCOMPATH)
      COMPNAME=$(basename $WCOMPATH)
      COMPILER=${COMPDIR}/${COMPNAME}
      ;;
    --with-options*)
      KOKKOS_OPT="${key#*=}"
      ;;
    --gcc-toolchain*)
      KOKKOS_GCC_TOOLCHAIN="${key#*=}"
      ;;
    --help)
      echo "Kokkos configure options:"
      echo ""
      echo "--kokkos-path=/Path/To/Kokkos:        Path to the Kokkos root directory."
      echo "--qthreads-path=/Path/To/Qthreads:    Path to Qthreads install directory."
      echo "                                        Overrides path given by --with-qthreads."
      echo "--prefix=/Install/Path:               Path to install the Kokkos library."
      echo ""
      echo "--with-cuda[=/Path/To/Cuda]:          Enable Cuda and set path to Cuda Toolkit."
      echo "--with-openmp:                        Enable OpenMP backend."
      echo "--with-pthread:                       Enable Pthreads backend."
      echo "--with-serial:                        Enable Serial backend."
      echo "--with-qthreads[=/Path/To/Qthreads]:  Enable Qthreads backend."
      echo "--with-devices:                       Explicitly add a set of backends."
      echo ""
      echo "--arch=[OPT]:  Set target architectures. Options are:"
      echo "               [AMD]"
      echo "                 AMDAVX          = AMD CPU"
      echo "                 EPYC            = AMD EPYC Zen-Core CPU"
      echo "               [ARM]"
      echo "                 ARMv80          = ARMv8.0 Compatible CPU"
      echo "                 ARMv81          = ARMv8.1 Compatible CPU"
      echo "                 ARMv8-ThunderX  = ARMv8 Cavium ThunderX CPU"
      echo "                 ARMv8-TX2       = ARMv8 Cavium ThunderX2 CPU"
      echo "               [IBM]"
      echo "                 BGQ             = IBM Blue Gene Q"
      echo "                 Power7          = IBM POWER7 and POWER7+ CPUs"
      echo "                 Power8          = IBM POWER8 CPUs"
      echo "                 Power9          = IBM POWER9 CPUs"
      echo "               [Intel]"
      echo "                 WSM             = Intel Westmere CPUs"
      echo "                 SNB             = Intel Sandy/Ivy Bridge CPUs"
      echo "                 HSW             = Intel Haswell CPUs"
      echo "                 BDW             = Intel Broadwell Xeon E-class CPUs"
      echo "                 SKX             = Intel Sky Lake Xeon E-class HPC CPUs (AVX512)"
      echo "               [Intel Xeon Phi]"
      echo "                 KNC             = Intel Knights Corner Xeon Phi"
      echo "                 KNL             = Intel Knights Landing Xeon Phi"
      echo "               [NVIDIA]"
      echo "                 Kepler30        = NVIDIA Kepler generation CC 3.0"
      echo "                 Kepler32        = NVIDIA Kepler generation CC 3.2"
      echo "                 Kepler35        = NVIDIA Kepler generation CC 3.5"
      echo "                 Kepler37        = NVIDIA Kepler generation CC 3.7"
      echo "                 Maxwell50       = NVIDIA Maxwell generation CC 5.0"
      echo "                 Maxwell52       = NVIDIA Maxwell generation CC 5.2"
      echo "                 Maxwell53       = NVIDIA Maxwell generation CC 5.3"
      echo "                 Pascal60        = NVIDIA Pascal generation CC 6.0"
      echo "                 Pascal61        = NVIDIA Pascal generation CC 6.1"
      echo "                 Volta70         = NVIDIA Volta generation CC 7.0"
      echo "                 Volta72         = NVIDIA Volta generation CC 7.2"
      echo ""
      echo "--compiler=/Path/To/Compiler  Set the compiler."
      echo "--debug,-dbg:                 Enable Debugging."
      echo "--cxxflags=[FLAGS]            Overwrite CXXFLAGS for library build and test"
      echo "                                build.  This will still set certain required"
      echo "                                flags via KOKKOS_CXXFLAGS (such as -fopenmp,"
      echo "                                --std=c++11, etc.)."
      echo "--cxxstandard=[FLAGS]         Overwrite KOKKOS_CXX_STANDARD for library build and test"
      echo "                                c++11 (default), c++14, c++17, c++1y, c++1z, c++2a"
      echo "--ldflags=[FLAGS]             Overwrite LDFLAGS for library build and test"
      echo "                                build. This will still set certain required"
      echo "                                flags via KOKKOS_LDFLAGS (such as -fopenmp,"
      echo "                                -lpthread, etc.)."
      echo "--with-gtest=/Path/To/Gtest:  Set path to gtest.  (Used in unit and performance"
      echo "                                tests.)"
      echo "--with-hwloc=/Path/To/Hwloc:  Set path to hwloc library."
      echo "--with-memkind=/Path/To/MemKind:  Set path to memkind library."
      echo "--with-options=[OPT]:         Additional options to Kokkos:"
      echo "                                compiler_warnings"
      echo "                                aggressive_vectorization = add ivdep on loops"
      echo "                                disable_profiling = do not compile with profiling hooks"
      echo "                                "
      echo "--with-cuda-options=[OPT]:    Additional options to CUDA:"
      echo "                                force_uvm, use_ldg, enable_lambda, rdc"
      echo "--with-hpx-options=[OPT]:     Additional options to HPX:"
      echo "                                enable_async_dispatch"
      echo "--gcc-toolchain=/Path/To/GccRoot:  Set the gcc toolchain to use with clang (e.g. /usr)" 
      echo "--make-j=[NUM]:               DEPRECATED: call make with appropriate"
      echo "                                -j flag"
      exit 0
      ;;
    *)
      echo "warning: ignoring unknown option $key"
      ;;
  esac

  shift
done

if [ "$COMPILER" == "" ]; then
    COMPILER_CMD=
else
    COMPILER_CMD='-DCMAKE_CXX_COMPILER=$COMPILER'
fi

echo cmake $COMPILER_CMD  -DCMAKE_CXX_FLAGS="${CXXFLAGS}" -DCMAKE_INSTALL_PREFIX=${PREFIX} -DKokkos_DEVICES=$KOKKOS_DEVICES -DKokkos_ARCH=$KOKKOS_ARCH -DKokkos_ENABLE_TESTS=ON -DKokkos_ENABLE_EXAMPLES=${KOKKOS_DO_EXAMPLES} -DKokkos_OPTIONS=${KOKKOS_OPT} -DCMAKE_CXX_COMPILER=${COMPILER} -DCMAKE_VERBOSE_MAKEFILE=ON -DCMAKE_CXX_EXTENSIONS=OFF -DKokkos_CUDA_DIR=${CUDA_PATH} -DKokkos_CXX_STANDARD=${KOKKOS_CXX_STANDARD} ${KOKKOS_PATH} 
cmake $COMPILER_CMD  -DCMAKE_CXX_FLAGS="${CXXFLAGS}" -DCMAKE_INSTALL_PREFIX=${PREFIX} -DKokkos_DEVICES=$KOKKOS_DEVICES -DKokkos_ARCH=$KOKKOS_ARCH -DKokkos_ENABLE_TESTS=ON -DKokkos_ENABLE_EXAMPLES=${KOKKOS_DO_EXAMPLES} -DKokkos_OPTIONS=${KOKKOS_OPT} -DCMAKE_CXX_COMPILER=${COMPILER} -DCMAKE_VERBOSE_MAKEFILE=ON -DCMAKE_CXX_EXTENSIONS=OFF -DKokkos_CUDA_DIR=${CUDA_PATH} -DKokkos_CXX_STANDARD=${KOKKOS_CXX_STANDARD} ${KOKKOS_PATH} 
