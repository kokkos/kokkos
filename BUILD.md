![Kokkos](https://avatars2.githubusercontent.com/u/10199860?s=200&v=4)

# Building Kokkos

Detailed build instructions coming soon!

# Kokkos Keyword Listing
* Kokkos_ARCH
    * Optimize for specific host architecture. Options are:, NONE, AMDAVX, ARMV80, ARMV81, ARMV8_THUNDERX, ARMV8_TX2, WSM, SNB, HSW, BDW, SKX, KNC, KNL, BGQ, POWER7, POWER8, POWER9, KEPLER, KEPLER30, KEPLER32, KEPLER35, KEPLER37, MAXWELL, MAXWELL50, MAXWELL52, MAXWELL53, PASCAL60, PASCAL61, VOLTA70, VOLTA72, TURING75, RYZEN, EPYC, KAVERI, CARRIZO, FIJI, VEGA, GFX901
    * STRING Default: NONE
* Kokkos_ARCH_AMDAVX
    * Whether to optimize for the AMDAVX architecture
    * BOOL Default: OFF
* Kokkos_ARCH_ARMV80
    * Whether to optimize for the ARMV80 architecture
    * BOOL Default: OFF
* Kokkos_ARCH_ARMV81
    * Whether to optimize for the ARMV81 architecture
    * BOOL Default: OFF
* Kokkos_ARCH_ARMV8_THUNDERX
    * Whether to optimize for the ARMV8_THUNDERX architecture
    * BOOL Default: OFF
* Kokkos_ARCH_ARMV8_TX2
    * Whether to optimize for the ARMV8_TX2 architecture
    * BOOL Default: OFF
* Kokkos_ARCH_BDW
    * Whether to optimize for the BDW architecture
    * BOOL Default: OFF
* Kokkos_ARCH_BGQ
    * Whether to optimize for the BGQ architecture
    * BOOL Default: OFF
* Kokkos_ARCH_CARRIZO
    * Whether to optimize for the CARRIZO architecture
    * BOOL Default: OFF
* Kokkos_ARCH_EPYC
    * Whether to optimize for the EPYC architecture
    * BOOL Default: OFF
* Kokkos_ARCH_FIJI
    * Whether to optimize for the FIJI architecture
    * BOOL Default: OFF
* Kokkos_ARCH_GFX901
    * Whether to optimize for the GFX901 architecture
    * BOOL Default: OFF
* Kokkos_ARCH_HSW
    * Whether to optimize for the HSW architecture
    * BOOL Default: OFF
* Kokkos_ARCH_KAVERI
    * Whether to optimize for the KAVERI architecture
    * BOOL Default: OFF
* Kokkos_ARCH_KEPLER
    * Whether to optimize for the KEPLER architecture
    * BOOL Default: OFF
* Kokkos_ARCH_KEPLER30
    * Whether to optimize for the KEPLER30 architecture
    * BOOL Default: OFF
* Kokkos_ARCH_KEPLER32
    * Whether to optimize for the KEPLER32 architecture
    * BOOL Default: OFF
* Kokkos_ARCH_KEPLER35
    * Whether to optimize for the KEPLER35 architecture
    * BOOL Default: OFF
* Kokkos_ARCH_KEPLER37
    * Whether to optimize for the KEPLER37 architecture
    * BOOL Default: OFF
* Kokkos_ARCH_KNC
    * Whether to optimize for the KNC architecture
    * BOOL Default: OFF
* Kokkos_ARCH_KNL
    * Whether to optimize for the KNL architecture
    * BOOL Default: OFF
* Kokkos_ARCH_MAXWELL
    * Whether to optimize for the MAXWELL architecture
    * BOOL Default: OFF
* Kokkos_ARCH_MAXWELL50
    * Whether to optimize for the MAXWELL50 architecture
    * BOOL Default: OFF
* Kokkos_ARCH_MAXWELL52
    * Whether to optimize for the MAXWELL52 architecture
    * BOOL Default: OFF
* Kokkos_ARCH_MAXWELL53
    * Whether to optimize for the MAXWELL53 architecture
    * BOOL Default: OFF
* Kokkos_ARCH_NONE
    * optimize for architecture NONE
    * BOOL Default: ON
* Kokkos_ARCH_PASCAL60
    * Whether to optimize for the PASCAL60 architecture
    * BOOL Default: OFF
* Kokkos_ARCH_PASCAL61
    * Whether to optimize for the PASCAL61 architecture
    * BOOL Default: OFF
* Kokkos_ARCH_POWER7
    * Whether to optimize for the POWER7 architecture
    * BOOL Default: OFF
* Kokkos_ARCH_POWER8
    * Whether to optimize for the POWER8 architecture
    * BOOL Default: OFF
* Kokkos_ARCH_POWER9
    * Whether to optimize for the POWER9 architecture
    * BOOL Default: OFF
* Kokkos_ARCH_RYZEN
    * Whether to optimize for the RYZEN architecture
    * BOOL Default: OFF
* Kokkos_ARCH_SKX
    * Whether to optimize for the SKX architecture
    * BOOL Default: OFF
* Kokkos_ARCH_SNB
    * Whether to optimize for the SNB architecture
    * BOOL Default: OFF
* Kokkos_ARCH_TURING75
    * Whether to optimize for the TURING75 architecture
    * BOOL Default: OFF
* Kokkos_ARCH_VEGA
    * Whether to optimize for the VEGA architecture
    * BOOL Default: OFF
* Kokkos_ARCH_VOLTA70
    * Whether to optimize for the VOLTA70 architecture
    * BOOL Default: OFF
* Kokkos_ARCH_VOLTA72
    * Whether to optimize for the VOLTA72 architecture
    * BOOL Default: OFF
* Kokkos_ARCH_WSM
    * Whether to optimize for the WSM architecture
    * BOOL Default: OFF
* Kokkos_BINARY_DIR
    * Value Computed by CMake
    * STATIC Default: /Users/jjwilke/Scratch/kokkos/default
* Kokkos_CUDA_DIR
    * Location of CUDA library
    * PATH Default: 
* Kokkos_CXX_STANDARD
    * The C++ standard for Kokkos to use: c++11, c++14, or c++17
    * STRING Default: 
* Kokkos_DEVICES
    * A list of devices to enable
    * STRING Default: SERIAL
* Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION
    * Whether to aggressively vectorize loops
    * BOOL Default: OFF
* Kokkos_ENABLE_COMPILER_WARNINGS
    * Whether to print all compiler warnings
    * BOOL Default: OFF
* Kokkos_ENABLE_CUDA
    * Whether to build CUDA backend
    * BOOL Default: OFF
* Kokkos_ENABLE_CUDA_LAMBDA
    * Whether to activate experimental laambda features
    * BOOL Default: OFF
* Kokkos_ENABLE_CUDA_LDG_INTRINSIC
    * Whether to use CUDA LDG intrinsics
    * BOOL Default: OFF
* Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE
    * Whether to enable relocatable device code (RDC) for CUDA
    * BOOL Default: OFF
* Kokkos_ENABLE_CUDA_UVM
    * Whether to enable unified virtual memory (UVM) for CUDA
    * BOOL Default: OFF
* Kokkos_ENABLE_DEBUG
    * Whether to activate extra debug features - may increase compiletimes
    * BOOL Default: OFF
* Kokkos_ENABLE_DEBUG_BOUNDS_CHECK
    * Whether to use bounds checking - will increase runtime
    * BOOL Default: OFF
* Kokkos_ENABLE_DEBUG_DUALVIEW_MODIFY_CHECK
    * Debug check on dual views
    * BOOL Default: OFF
* Kokkos_ENABLE_DEPRECATED_CODE
    * Whether to enable deprecated code
    * BOOL Default: OFF
* Kokkos_ENABLE_EXAMPLES
    * Whether to build OpenMP  backend
    * BOOL Default: OFF
* Kokkos_ENABLE_EXPLICIT_INSTANTIATION
    * Whether to explicitly instantiate certain types to lower futurecompile times
    * BOOL Default: OFF
* Kokkos_ENABLE_HPX
    * Whether to enable the HPX library
    * BOOL Default: OFF
* Kokkos_ENABLE_HPX_ASYNC_DISPATCH
    * Whether HPX supports asynchronous dispath
    * BOOL Default: OFF
* Kokkos_ENABLE_HWLOC
    * Whether to enable the HWLOC library
    * BOOL Default: Off
* Kokkos_ENABLE_LIBNUMA
    * Whether to enable the LIBNUMA library
    * BOOL Default: Off
* Kokkos_ENABLE_MEMKIND
    * Whether to enable the MEMKIND library
    * BOOL Default: Off
* Kokkos_ENABLE_OPENMP
    * Whether to build OpenMP backend
    * BOOL Default: OFF
* Kokkos_ENABLE_PROFILING
    * Whether to create bindings for profiling tools
    * BOOL Default: ON
* Kokkos_ENABLE_PROFILING_LOAD_PRINT
    * Whether to print information about which profiling tools gotloaded
    * BOOL Default: OFF
* Kokkos_ENABLE_PTHREAD
    * Whether to build Pthread backend
    * BOOL Default: OFF
* Kokkos_ENABLE_QTHREAD
    * Whether to enable the QTHREAD library
    * BOOL Default: OFF
* Kokkos_ENABLE_ROCM
    * Whether to build AMD ROCm backend
    * BOOL Default: OFF
* Kokkos_ENABLE_SERIAL
    * Whether to build serial  backend
    * BOOL Default: ON
* Kokkos_ENABLE_TESTS
    * Whether to build serial  backend
    * BOOL Default: OFF
* Kokkos_HPX_DIR
    * Location of HPX library
    * PATH Default: 
* Kokkos_HWLOC_DIR
    * Location of HWLOC library
    * PATH Default: 
* Kokkos_LIBNUMA_DIR
    * Location of LIBNUMA library
    * PATH Default: 
* Kokkos_MEMKIND_DIR
    * Location of MEMKIND library
    * PATH Default: 
* Kokkos_OPTIONS
    * A list of options to enable
    * STRING Default: 
* Kokkos_QTHREAD_DIR
    * Location of QTHREAD library
    * PATH Default: 
* Kokkos_SEPARATE_LIBS
    * whether to build libkokkos or libkokkoscontainers, etc
    * BOOL Default: OFF
* Kokkos_SEPARATE_TESTS
    * Provide unit test targets with finer granularity.
    * BOOL Default: OFF



##### [LICENSE](https://github.com/sstsimulator/sst-core/blob/devel/LICENSE)

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Under the terms of Contract DE-NA0003525 with NTESS,
the U.S. Government retains certain rights in this software.
