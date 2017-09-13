########################## COMPILER AND FEATURE CHECKS #########################

# TODO: We are assuming that nvcc_wrapper is using g++ as the host compiler.
#       Should we allow the user the option to change this?  The host compiler
#       for nvcc_wrapper can be set via the NVCC_WRAPPER_DEFAULT_COMPILER
#       environment variable or by passing a different host compiler with the
#       -ccbin flag.

# TODO: Fully add CUDA support for Clang.
include(${CMAKE_SOURCE_DIR}/cmake/kokkos_functions.cmake)
set_kokkos_cxx_compiler()

set_kokkos_compiler_standard()

########################## COMPILER AND FEATURE CHECKS #########################
# List of possible host architectures.
list(APPEND KOKKOS_HOST_ARCH_LIST
     None            # No architecture optimization
     AMDAVX          # AMD chip
     ARMv80          # ARMv8.0 Compatible CPU
     ARMv81          # ARMv8.1 Compatible CPU
     ARMv8-ThunderX  # ARMv8 Cavium ThunderX CPU
     SNB             # Intel Sandy/Ivy Bridge CPUs
     HSW             # Intel Haswell CPUs
     BDW             # Intel Broadwell Xeon E-class CPUs
     SKX             # Intel Sky Lake Xeon E-class HPC CPUs (AVX512)
     KNC             # Intel Knights Corner Xeon Phi
     KNL             # Intel Knights Landing Xeon Phi
     BGQ             # IBM Blue Gene Q
     Power7          # IBM POWER7 CPUs
     Power8          # IBM POWER8 CPUs
     Power9          # IBM POWER9 CPUs
    )

# KOKKOS_HOST_ARCH must be defined previoiusly
#set_property(CACHE KOKKOS_HOST_ARCH PROPERTY STRINGS ${KOKKOS_HOST_ARCH_LIST})

# List of possible host architectures.
list(APPEND KOKKOS_DEVICES_LIST
    CUDA          # NVIDIA GPU -- see below
    OPENMP        # OpenMP
    PTHREAD       # pthread
    QTHREADS      # qthreads
    SERIAL        # serial
    )


# List of possible GPU architectures.
list(APPEND KOKKOS_GPU_ARCH_LIST
     None            # No architecture optimization
     Kepler          # NVIDIA Kepler default (generation CC 3.5)
     Kepler30        # NVIDIA Kepler generation CC 3.0
     Kepler32        # NVIDIA Kepler generation CC 3.2
     Kepler35        # NVIDIA Kepler generation CC 3.5
     Kepler37        # NVIDIA Kepler generation CC 3.7
     Maxwell         # NVIDIA Maxwell default (generation CC 5.0)
     Maxwell50       # NVIDIA Maxwell generation CC 5.0
     Maxwell52       # NVIDIA Maxwell generation CC 5.2
     Maxwell53       # NVIDIA Maxwell generation CC 5.3
     Pascal60        # NVIDIA Pascal generation CC 6.0
     Pascal61        # NVIDIA Pascal generation CC 6.1
    )

# KOKKOS_GPU_ARCH must be defined previoiusly
set_property(CACHE KOKKOS_GPU_ARCH PROPERTY STRINGS ${KOKKOS_GPU_ARCH_LIST})


######################### INITIALIZE INTERNAL VARIABLES ########################

# Add Kokkos' modules to CMake's module path.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${Kokkos_SOURCE_DIR}/cmake/Modules/")

# Start with all global variables set to false.  This guarantees correct
# results with changes and multiple configures.
set(KOKKOS_HAVE_CUDA OFF CACHE INTERNAL "")
set(KOKKOS_USE_CUDA_UVM OFF CACHE INTERNAL "")
set(KOKKOS_HAVE_CUDA_RDC OFF CACHE INTERNAL "")
set(KOKKOS_HAVE_CUDA_LAMBDA OFF CACHE INTERNAL "")
set(KOKKOS_CUDA_CLANG_WORKAROUND OFF CACHE INTERNAL "")
set(KOKKOS_HAVE_OPENMP OFF CACHE INTERNAL "")
set(KOKKOS_HAVE_PTHREAD OFF CACHE INTERNAL "")
set(KOKKOS_HAVE_QTHREADS OFF CACHE INTERNAL "")
set(KOKKOS_HAVE_SERIAL OFF CACHE INTERNAL "")
set(KOKKOS_HAVE_HWLOC OFF CACHE INTERNAL "")
set(KOKKOS_ENABLE_HBWSPACE OFF CACHE INTERNAL "")
set(KOKKOS_HAVE_DEBUG OFF CACHE INTERNAL "")
set(KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK OFF CACHE INTERNAL "")
set(KOKKOS_ENABLE_ISA_X86_64 OFF CACHE INTERNAL "")
set(KOKKOS_ENABLE_ISA_KNC OFF CACHE INTERNAL "")
set(KOKKOS_ENABLE_ISA_POWERPCLE OFF CACHE INTERNAL "")
set(KOKKOS_ARCH_ARMV80 OFF CACHE INTERNAL "")
set(KOKKOS_ARCH_ARMV81 OFF CACHE INTERNAL "")
set(KOKKOS_ARCH_ARMV8_THUNDERX OFF CACHE INTERNAL "")
set(KOKKOS_ARCH_AVX OFF CACHE INTERNAL "")
set(KOKKOS_ARCH_AVX2 OFF CACHE INTERNAL "")
set(KOKKOS_ARCH_AVX512MIC OFF CACHE INTERNAL "")
set(KOKKOS_ARCH_AVX512XEON OFF CACHE INTERNAL "")
set(KOKKOS_ARCH_KNC OFF CACHE INTERNAL "")
set(KOKKOS_ARCH_POWER8 OFF CACHE INTERNAL "")
set(KOKKOS_ARCH_POWER9 OFF CACHE INTERNAL "")
set(KOKKOS_ARCH_KEPLER OFF CACHE INTERNAL "")
set(KOKKOS_ARCH_KEPLER30 OFF CACHE INTERNAL "")
set(KOKKOS_ARCH_KEPLER32 OFF CACHE INTERNAL "")
set(KOKKOS_ARCH_KEPLER35 OFF CACHE INTERNAL "")
set(KOKKOS_ARCH_KEPLER37 OFF CACHE INTERNAL "")
set(KOKKOS_ARCH_MAXWELL OFF CACHE INTERNAL "")
set(KOKKOS_ARCH_MAXWELL50 OFF CACHE INTERNAL "")
set(KOKKOS_ARCH_MAXWELL52 OFF CACHE INTERNAL "")
set(KOKKOS_ARCH_MAXWELL53 OFF CACHE INTERNAL "")
set(KOKKOS_ARCH_PASCAL OFF CACHE INTERNAL "")
set(KOKKOS_ARCH_PASCAL60 OFF CACHE INTERNAL "")
set(KOKKOS_ARCH_PASCAL61 OFF CACHE INTERNAL "")


# Set which Kokkos backend to use.
set(KOKKOS_ENABLE_CUDA OFF CACHE BOOL "Use Kokkos CUDA backend")
set(KOKKOS_ENABLE_OPENMP ON CACHE BOOL "Use Kokkos OpenMP backend")
set(KOKKOS_ENABLE_PTHREAD OFF CACHE BOOL "Use Kokkos Pthreads backend")
set(KOKKOS_ENABLE_QTHREADS OFF CACHE BOOL "Use Kokkos Qthreads backend")
set(KOKKOS_ENABLE_SERIAL ON CACHE BOOL "Use Kokkos Serial backend")

# Can have multiple devices 
set(KOKKOS_DEVICESl)
foreach(devopt ${KOKKOS_DEVICES_LIST})
  if (${KOKKOS_ENABLE_${devopt}}) 
    list(APPEND KOKKOS_DEVICESl ${devopt})
  endif ()
endforeach()
# List needs to be comma-delmitted
string(REPLACE ";" "," KOKKOS_DEVICES "${KOKKOS_DEVICESl}")

# Setting this variable to a value other than "None" can improve host
# performance by turning on architecture specific code.
set(KOKKOS_HOST_ARCH "None" CACHE STRING "Optimize for specific host architecture.")

# Ensure that KOKKOS_HOST_ARCH is in the ARCH_LIST
list(FIND KOKKOS_HOST_ARCH_LIST ${KOKKOS_HOST_ARCH} indx)
if (indx EQUAL -1)
  message(FATAL_ERROR "${KOKKOS_HOST_ARCH} is not an accepted host")
  #message(WARNING "${KOKKOS_HOST_ARCH} is not an accepted host")
endif ()

# KOKKOS_SETTINGS uses KOKKOS_ARCH
set(KOKKOS_ARCH ${KOKKOS_HOST_ARCH})

# Setting this variable to a value other than "None" can improve GPU
# performance by turning on architecture specific code.
set(KOKKOS_GPU_ARCH "None" CACHE STRING "Optimize for specific GPU architecture.")

set(KOKKOS_SEPARATE_LIBS OFF CACHE BOOL "OFF = kokkos.  ON = kokkoscore, kokkoscontainers, and kokkosalgorithms.")

# Enable hwloc library.
set(KOKKOS_ENABLE_HWLOC OFF CACHE BOOL "Enable hwloc for better process placement.")
set(KOKKOS_HWLOC_DIR "" CACHE PATH "Location of hwloc library.")

# Enable memkind library.
set(KOKKOS_ENABLE_MEMKIND OFF CACHE BOOL "Enable memkind.")
set(KOKKOS_MEMKIND_DIR "" CACHE PATH "Location of memkind library.")

set(KOKKOS_ENABLE_LIBRT OFF CACHE BOOL "Enable librt for more precise timer.")

# Enable debugging.
set(KOKKOS_DEBUG OFF CACHE BOOL "Enable debugging in Kokkos.")

# Enable profiling.
set(KOKKOS_ENABLE_PROFILING ON CACHE BOOL "Enable profiling.")

# Enable aggressive vectorization.
set(KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION OFF CACHE BOOL "Enable aggressive vectorization.")

# Qthreads options.
set(KOKKOS_QTHREADS_DIR "" CACHE PATH "Location of Qthreads library.")

# CUDA options.
set(KOKKOS_CUDA_DIR "" CACHE PATH "Location of CUDA library.  Defaults to where nvcc installed.")
set(KOKKOS_ENABLE_CUDA_LDG_INTRINSIC OFF CACHE BOOL "Enable CUDA LDG.")
set(KOKKOS_ENABLE_CUDA_UVM OFF CACHE BOOL "Enable CUDA unified virtual memory.")
set(KOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE OFF CACHE BOOL "Enable relocatable device code for CUDA.")
set(KOKKOS_ENABLE_CUDA_LAMBDA ON CACHE BOOL "Enable lambdas for CUDA.")


# Set the KOKKOS_SETTINGS String -- this is the primary communication with the
# makefile configuration

# These are consistent with generate_makefile.bash:
#set(KOKKOS_SETVARS "SRC_PATH;PATH;ARCH;DEBUG;OPTIONS;USE_TPLS;OPT;CUDA_OPT")
#set(OTHER_SETVARS "CXX;CXXFLAGS;CUDA_PATH;LDFLAGS;GTEST_PATH;")

# This is the union of original cmake/kokkos.cmake and generate_makefile.bash
#    These are the diffs: OPTIONS;USE_TPLS;OPT;CUDA_OPT
set(KOKKOS_SETVARS "ARCH;DEVICES;DEBUG")
set(OTHER_SETVARS "CXX;CXXFLAGS;CUDA_PATH;LDFLAGS")
set(OTHER_SETVARS ${OTHER_SETVARS};"GTEST_PATH;HWLOC_PATH;MEMKIND_PATH;QTHREADS_PATH")

set(KOKKOS_SETTINGS "KOKKOS_SRC_PATH=${CMAKE_SOURCE_DIR}")
set(KOKKOS_SETTINGS "${KOKKOS_SETTINGS} KOKKOS_PATH=${CMAKE_SOURCE_DIR}")
foreach(kvar ${KOKKOS_SETVARS})
  if(DEFINED KOKKOS_${kvar})
    set(KOKKOS_SETTINGS "${KOKKOS_SETTINGS} KOKKOS_${kvar}=${KOKKOS_${kvar}}")
  endif()
endforeach()
foreach(ovar ${OTHER_SETVARS})
  if(DEFINED ${ovar})
    set(KOKKOS_SETTINGS "${KOKKOS_SETTINGS} ${ovar}=${${ovar}}")
  endif()
endforeach()
set(KOKKOS_SETTINGS "KOKKOS_SETTINGS=\"${KOKKOS_SETTINGS}\"")
