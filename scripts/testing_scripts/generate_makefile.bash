#!/bin/bash

KOKKOS_DEVICES=""

while [[ $# > 0 ]]
do
  key="$1"

  case $key in
    --kokkos-path*)
      KOKKOS_PATH="${key#*=}"
      ;;
    --hpx-path*)
      HPX_PATH="${key#*=}"
      ;;
    --prefix*)
      PREFIX="${key#*=}"
      ;;
    --with-cuda)
      KOKKOS_DEVICES="${KOKKOS_DEVICES},Cuda"
      CUDA_PATH_NVCC=$(command -v nvcc)
      CUDA_PATH=${CUDA_PATH_NVCC%/bin/nvcc}
      ;;
    # Catch this before '--with-cuda*'
    --with-cuda-options*)
      KOKKOS_CUDA_OPT="${key#*=}"
      ;;
    --with-cuda*)
      KOKKOS_DEVICES="${KOKKOS_DEVICES},Cuda"
      CUDA_PATH="${key#*=}"
      ;;
    --with-openmp)
      KOKKOS_DEVICES="${KOKKOS_DEVICES},OpenMP"
      ;;
    --with-pthread)
      KOKKOS_DEVICES="${KOKKOS_DEVICES},Pthread"
      ;;
    --with-serial)
      KOKKOS_DEVICES="${KOKKOS_DEVICES},Serial"
      ;;
    --with-hpx-options*)
      KOKKOS_HPX_OPT="${key#*=}"
      ;;
    --with-hpx*)
      KOKKOS_DEVICES="${KOKKOS_DEVICES},HPX"
      if [ -z "$HPX_PATH" ]; then
        HPX_PATH="${key#*=}"
      fi
      ;;
    --with-devices*)
      DEVICES="${key#*=}"
      KOKKOS_DEVICES="${KOKKOS_DEVICES},${DEVICES}"
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
      echo "--prefix=/Install/Path:               Path to install the Kokkos library."
      echo ""
      echo "--with-cuda[=/Path/To/Cuda]:          Enable Cuda and set path to Cuda Toolkit."
      echo "--with-openmp:                        Enable OpenMP backend."
      echo "--with-pthread:                       Enable Pthreads backend."
      echo "--with-serial:                        Enable Serial backend."
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

# Remove leading ',' from KOKKOS_DEVICES.
KOKKOS_DEVICES=$(echo $KOKKOS_DEVICES | sed 's/^,//')

# If KOKKOS_PATH undefined, assume parent dir of this script is the KOKKOS_PATH.
if [ -z "$KOKKOS_PATH" ]; then
  KOKKOS_PATH=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd ../../ && pwd )
else
  # Ensure KOKKOS_PATH is abs path
  KOKKOS_PATH=$( cd $KOKKOS_PATH && pwd )
fi

if [ "${KOKKOS_PATH}"  = "${PWD}" ] || [ "${KOKKOS_PATH}"  = "${PWD}/" ]; then
  echo "Running generate_makefile.bash in the Kokkos root directory is not allowed"
  exit
fi

KOKKOS_SRC_PATH=${KOKKOS_PATH}

KOKKOS_SETTINGS="KOKKOS_SRC_PATH=${KOKKOS_SRC_PATH}"

# The double [[  ]] in the elif branch is not a typo
if [ ${#COMPILER} -gt 0 ]; then
  KOKKOS_SETTINGS="${KOKKOS_SETTINGS} CXX=${COMPILER}"
elif
   [ ${#COMPILER} -eq 0 ] && [[ ${KOKKOS_DEVICES} =~ .*Cuda.* ]]; then
  COMPILER="${KOKKOS_PATH}/bin/nvcc_wrapper"
  KOKKOS_SETTINGS="${KOKKOS_SETTINGS} CXX=${COMPILER}"   
fi

if [ ${#KOKKOS_DEVICES} -gt 0 ]; then
  KOKKOS_SETTINGS="${KOKKOS_SETTINGS} KOKKOS_DEVICES=${KOKKOS_DEVICES}"
fi

if [ ${#KOKKOS_ARCH} -gt 0 ]; then
  KOKKOS_SETTINGS="${KOKKOS_SETTINGS} KOKKOS_ARCH=${KOKKOS_ARCH}"
fi

if [ ${#KOKKOS_DEBUG} -gt 0 ]; then
  KOKKOS_SETTINGS="${KOKKOS_SETTINGS} KOKKOS_DEBUG=${KOKKOS_DEBUG}"
fi

if [ ${#CUDA_PATH} -gt 0 ]; then
  KOKKOS_SETTINGS="${KOKKOS_SETTINGS} CUDA_PATH=${CUDA_PATH}"
fi

if [ ${#CXXFLAGS} -gt 0 ]; then
  KOKKOS_SETTINGS="${KOKKOS_SETTINGS} CXXFLAGS=\"${CXXFLAGS}\""
fi

if [ ${#KOKKOS_CXX_STANDARD} -gt 0 ]; then
  KOKKOS_SETTINGS="${KOKKOS_SETTINGS} KOKKOS_CXX_STANDARD=\"${KOKKOS_CXX_STANDARD}\""
fi

if [ ${#LDFLAGS} -gt 0 ]; then
  KOKKOS_SETTINGS="${KOKKOS_SETTINGS} LDFLAGS=\"${LDFLAGS}\""
fi

if [ ${#GTEST_PATH} -gt 0 ]; then
  KOKKOS_SETTINGS="${KOKKOS_SETTINGS} GTEST_PATH=${GTEST_PATH}"
else
  GTEST_PATH=${KOKKOS_PATH}/tpls/gtest
  KOKKOS_SETTINGS="${KOKKOS_SETTINGS} GTEST_PATH=${GTEST_PATH}"
fi

if [ ${#HWLOC_PATH} -gt 0 ]; then
  KOKKOS_SETTINGS="${KOKKOS_SETTINGS} HWLOC_PATH=${HWLOC_PATH}"
  KOKKOS_USE_TPLS="${KOKKOS_USE_TPLS},hwloc"
fi

if [ ${#MEMKIND_PATH} -gt 0 ]; then
  KOKKOS_SETTINGS="${KOKKOS_SETTINGS} MEMKIND_PATH=${MEMKIND_PATH}" 
  KOKKOS_USE_TPLS="${KOKKOS_USE_TPLS},experimental_memkind"
fi

if [ ${#KOKKOS_USE_TPLS} -gt 0 ]; then
  KOKKOS_SETTINGS="${KOKKOS_SETTINGS} KOKKOS_USE_TPLS=${KOKKOS_USE_TPLS}"
fi

if [ ${#QTHREADS_PATH} -gt 0 ]; then
  KOKKOS_SETTINGS="${KOKKOS_SETTINGS} QTHREADS_PATH=${QTHREADS_PATH}"
fi

if [ ${#HPX_PATH} -gt 0 ]; then
    KOKKOS_SETTINGS="${KOKKOS_SETTINGS} HPX_PATH=${HPX_PATH}"
fi

if [ ${#KOKKOS_OPT} -gt 0 ]; then
  KOKKOS_SETTINGS="${KOKKOS_SETTINGS} KOKKOS_OPTIONS=${KOKKOS_OPT}"
fi

if [ ${#KOKKOS_CUDA_OPT} -gt 0 ]; then
  KOKKOS_SETTINGS="${KOKKOS_SETTINGS} KOKKOS_CUDA_OPTIONS=${KOKKOS_CUDA_OPT}"
fi

if [ ${#KOKKOS_HPX_OPT} -gt 0 ]; then
    KOKKOS_SETTINGS="${KOKKOS_SETTINGS} KOKKOS_HPX_OPTIONS=${KOKKOS_HPX_OPT}"
fi

if [ ${#KOKKOS_GCC_TOOLCHAIN} -gt 0 ]; then
  KOKKOS_SETTINGS="${KOKKOS_SETTINGS} KOKKOS_INTERNAL_GCC_TOOLCHAIN=${KOKKOS_GCC_TOOLCHAIN}"
fi

KOKKOS_SETTINGS_NO_KOKKOS_PATH="${KOKKOS_SETTINGS}"


gen_makefile=Makefile.kokkos
echo "#Makefile to satisfy existence of target kokkos-clean before installing the library" > install/${gen_makefile}
echo "kokkos-clean:" >> install/${gen_makefile}
echo "" >> install/${gen_makefile}
mkdir -p core
mkdir -p core/unit_test
mkdir -p core/perf_test
mkdir -p containers
mkdir -p containers/unit_tests
mkdir -p containers/performance_tests
mkdir -p algorithms
mkdir -p algorithms/unit_tests
mkdir -p algorithms/performance_tests
mkdir -p example
mkdir -p example/fixture
mkdir -p example/feint
mkdir -p example/fenl
mkdir -p example/make_buildlink
mkdir -p example/tutorial

KOKKOS_SETTINGS="${KOKKOS_SETTINGS_NO_KOKKOS_PATH} KOKKOS_PATH=${KOKKOS_PATH}"

# Generate subdirectory makefiles.
echo "KOKKOS_SETTINGS=${KOKKOS_SETTINGS}" > core/unit_test/Makefile
echo "" >> core/unit_test/Makefile
echo "all:" >> core/unit_test/Makefile
echo -e "\t\$(MAKE) -f ${KOKKOS_PATH}/core/unit_test/Makefile ${KOKKOS_SETTINGS}" >> core/unit_test/Makefile
echo "" >> core/unit_test/Makefile
echo "test: all" >> core/unit_test/Makefile
echo -e "\t\$(MAKE) -f ${KOKKOS_PATH}/core/unit_test/Makefile ${KOKKOS_SETTINGS} test" >> core/unit_test/Makefile
echo "" >> core/unit_test/Makefile
echo "clean:" >> core/unit_test/Makefile
echo -e "\t\$(MAKE) -f ${KOKKOS_PATH}/core/unit_test/Makefile ${KOKKOS_SETTINGS} clean" >> core/unit_test/Makefile

echo "KOKKOS_SETTINGS=${KOKKOS_SETTINGS}" > core/perf_test/Makefile
echo "" >> core/perf_test/Makefile
echo "all:" >> core/perf_test/Makefile
echo -e "\t\$(MAKE) -f ${KOKKOS_PATH}/core/perf_test/Makefile ${KOKKOS_SETTINGS}" >> core/perf_test/Makefile
echo "" >> core/perf_test/Makefile
echo "test: all" >> core/perf_test/Makefile
echo -e "\t\$(MAKE) -f ${KOKKOS_PATH}/core/perf_test/Makefile ${KOKKOS_SETTINGS} test" >> core/perf_test/Makefile
echo "" >> core/perf_test/Makefile
echo "clean:" >> core/perf_test/Makefile
echo -e "\t\$(MAKE) -f ${KOKKOS_PATH}/core/perf_test/Makefile ${KOKKOS_SETTINGS} clean" >> core/perf_test/Makefile

echo "KOKKOS_SETTINGS=${KOKKOS_SETTINGS}" > containers/unit_tests/Makefile
echo "" >> containers/unit_tests/Makefile
echo "all:" >> containers/unit_tests/Makefile
echo -e "\t\$(MAKE) -f ${KOKKOS_PATH}/containers/unit_tests/Makefile ${KOKKOS_SETTINGS}" >> containers/unit_tests/Makefile
echo "" >> containers/unit_tests/Makefile
echo "test: all" >> containers/unit_tests/Makefile
echo -e "\t\$(MAKE) -f ${KOKKOS_PATH}/containers/unit_tests/Makefile ${KOKKOS_SETTINGS} test" >> containers/unit_tests/Makefile
echo "" >> containers/unit_tests/Makefile
echo "clean:" >> containers/unit_tests/Makefile
echo -e "\t\$(MAKE) -f ${KOKKOS_PATH}/containers/unit_tests/Makefile ${KOKKOS_SETTINGS} clean" >> containers/unit_tests/Makefile

echo "KOKKOS_SETTINGS=${KOKKOS_SETTINGS}" > containers/performance_tests/Makefile
echo "" >> containers/performance_tests/Makefile
echo "all:" >> containers/performance_tests/Makefile
echo -e "\t\$(MAKE) -f ${KOKKOS_PATH}/containers/performance_tests/Makefile ${KOKKOS_SETTINGS}" >> containers/performance_tests/Makefile
echo "" >> containers/performance_tests/Makefile
echo "test: all" >> containers/performance_tests/Makefile
echo -e "\t\$(MAKE) -f ${KOKKOS_PATH}/containers/performance_tests/Makefile ${KOKKOS_SETTINGS} test" >> containers/performance_tests/Makefile
echo "" >> containers/performance_tests/Makefile
echo "clean:" >> containers/performance_tests/Makefile
echo -e "\t\$(MAKE) -f ${KOKKOS_PATH}/containers/performance_tests/Makefile ${KOKKOS_SETTINGS} clean" >> containers/performance_tests/Makefile

echo "KOKKOS_SETTINGS=${KOKKOS_SETTINGS}" > algorithms/unit_tests/Makefile
echo "" >> algorithms/unit_tests/Makefile
echo "all:" >> algorithms/unit_tests/Makefile
echo -e "\t\$(MAKE) -f ${KOKKOS_PATH}/algorithms/unit_tests/Makefile ${KOKKOS_SETTINGS}" >> algorithms/unit_tests/Makefile
echo "" >> algorithms/unit_tests/Makefile
echo "test: all" >> algorithms/unit_tests/Makefile
echo -e "\t\$(MAKE) -f ${KOKKOS_PATH}/algorithms/unit_tests/Makefile ${KOKKOS_SETTINGS} test" >> algorithms/unit_tests/Makefile
echo "" >> algorithms/unit_tests/Makefile
echo "clean:" >> algorithms/unit_tests/Makefile
echo -e "\t\$(MAKE) -f ${KOKKOS_PATH}/algorithms/unit_tests/Makefile ${KOKKOS_SETTINGS} clean" >> algorithms/unit_tests/Makefile

echo "KOKKOS_SETTINGS=${KOKKOS_SETTINGS}" > example/fixture/Makefile
echo "" >> example/fixture/Makefile
echo "all:" >> example/fixture/Makefile
echo -e "\t\$(MAKE) -f ${KOKKOS_PATH}/example/fixture/Makefile ${KOKKOS_SETTINGS}" >> example/fixture/Makefile
echo "" >> example/fixture/Makefile
echo "test: all" >> example/fixture/Makefile
echo -e "\t\$(MAKE) -f ${KOKKOS_PATH}/example/fixture/Makefile ${KOKKOS_SETTINGS} test" >> example/fixture/Makefile
echo "" >> example/fixture/Makefile
echo "clean:" >> example/fixture/Makefile
echo -e "\t\$(MAKE) -f ${KOKKOS_PATH}/example/fixture/Makefile ${KOKKOS_SETTINGS} clean" >> example/fixture/Makefile

echo "KOKKOS_SETTINGS=${KOKKOS_SETTINGS}" > example/feint/Makefile
echo "" >> example/feint/Makefile
echo "all:" >> example/feint/Makefile
echo -e "\t\$(MAKE) -f ${KOKKOS_PATH}/example/feint/Makefile ${KOKKOS_SETTINGS}" >> example/feint/Makefile
echo "" >> example/feint/Makefile
echo "test: all" >> example/feint/Makefile
echo -e "\t\$(MAKE) -f ${KOKKOS_PATH}/example/feint/Makefile ${KOKKOS_SETTINGS} test" >> example/feint/Makefile
echo "" >> example/feint/Makefile
echo "clean:" >> example/feint/Makefile
echo -e "\t\$(MAKE) -f ${KOKKOS_PATH}/example/feint/Makefile ${KOKKOS_SETTINGS} clean" >> example/feint/Makefile

echo "KOKKOS_SETTINGS=${KOKKOS_SETTINGS}" > example/fenl/Makefile
echo "" >> example/fenl/Makefile
echo "all:" >> example/fenl/Makefile
echo -e "\t\$(MAKE) -f ${KOKKOS_PATH}/example/fenl/Makefile ${KOKKOS_SETTINGS}" >> example/fenl/Makefile
echo "" >> example/fenl/Makefile
echo "test: all" >> example/fenl/Makefile
echo -e "\t\$(MAKE) -f ${KOKKOS_PATH}/example/fenl/Makefile ${KOKKOS_SETTINGS} test" >> example/fenl/Makefile
echo "" >> example/fenl/Makefile
echo "clean:" >> example/fenl/Makefile
echo -e "\t\$(MAKE) -f ${KOKKOS_PATH}/example/fenl/Makefile ${KOKKOS_SETTINGS} clean" >> example/fenl/Makefile

echo "KOKKOS_SETTINGS=${KOKKOS_SETTINGS}" > example/make_buildlink/Makefile
echo "" >> example/make_buildlink/Makefile
echo "build:" >> example/make_buildlink/Makefile
echo -e "\t\$(MAKE) -f ${KOKKOS_PATH}/example/make_buildlink/Makefile ${KOKKOS_SETTINGS} build" >> example/make_buildlink/Makefile
echo "" >> example/make_buildlink/Makefile
echo "test: build" >> example/make_buildlink/Makefile
echo -e "\t\$(MAKE) -f ${KOKKOS_PATH}/example/make_buildlink/Makefile ${KOKKOS_SETTINGS} test" >> example/make_buildlink/Makefile
echo "" >> example/make_buildlink/Makefile
echo "clean:" >> example/make_buildlink/Makefile
echo -e "\t\$(MAKE) -f ${KOKKOS_PATH}/example/make_buildlink/Makefile ${KOKKOS_SETTINGS} clean" >> example/make_buildlink/Makefile

echo "KOKKOS_SETTINGS=${KOKKOS_SETTINGS}" > example/tutorial/Makefile
echo "" >> example/tutorial/Makefile
echo "build:" >> example/tutorial/Makefile
echo -e "\t\$(MAKE) -f ${KOKKOS_PATH}/example/tutorial/Makefile KOKKOS_SETTINGS='${KOKKOS_SETTINGS}' KOKKOS_PATH=${KOKKOS_PATH} build">> example/tutorial/Makefile
echo "" >> example/tutorial/Makefile
echo "test: build" >> example/tutorial/Makefile
echo -e "\t\$(MAKE) -f ${KOKKOS_PATH}/example/tutorial/Makefile KOKKOS_SETTINGS='${KOKKOS_SETTINGS}' KOKKOS_PATH=${KOKKOS_PATH} test" >> example/tutorial/Makefile
echo "" >> example/tutorial/Makefile
echo "clean:" >> example/tutorial/Makefile
echo -e "\t\$(MAKE) -f ${KOKKOS_PATH}/example/tutorial/Makefile KOKKOS_SETTINGS='${KOKKOS_SETTINGS}' KOKKOS_PATH=${KOKKOS_PATH} clean" >> example/tutorial/Makefile

# Generate top level directory makefile.
echo "Generating Makefiles with options " ${KOKKOS_SETTINGS}
echo "KOKKOS_SETTINGS=${KOKKOS_SETTINGS}" > Makefile
echo "" >> Makefile
echo "build-test:" >> Makefile
echo -e "\t\$(MAKE) -C core/unit_test" >> Makefile
echo -e "\t\$(MAKE) -C core/perf_test" >> Makefile
echo -e "\t\$(MAKE) -C containers/unit_tests" >> Makefile
echo -e "\t\$(MAKE) -C containers/performance_tests" >> Makefile
echo -e "\t\$(MAKE) -C algorithms/unit_tests" >> Makefile
echo "" >> Makefile
echo "test: build-test" >> Makefile
echo -e "\t\$(MAKE) -C core/unit_test test" >> Makefile
echo -e "\t\$(MAKE) -C core/perf_test test" >> Makefile
echo -e "\t\$(MAKE) -C containers/unit_tests test" >> Makefile
echo -e "\t\$(MAKE) -C containers/performance_tests test" >> Makefile
echo -e "\t\$(MAKE) -C algorithms/unit_tests test" >> Makefile
echo "" >> Makefile
echo "unit-tests-only:" >> Makefile
echo -e "\t\$(MAKE) -C core/unit_test test" >> Makefile
echo -e "\t\$(MAKE) -C containers/unit_tests test" >> Makefile
echo -e "\t\$(MAKE) -C algorithms/unit_tests test" >> Makefile
echo "" >> Makefile

echo "clean:" >> Makefile
echo -e "\t\$(MAKE) -C core/unit_test clean" >> Makefile
echo -e "\t\$(MAKE) -C core/perf_test clean" >> Makefile
echo -e "\t\$(MAKE) -C containers/unit_tests clean" >> Makefile
echo -e "\t\$(MAKE) -C containers/performance_tests clean" >> Makefile
echo -e "\t\$(MAKE) -C algorithms/unit_tests clean" >> Makefile

