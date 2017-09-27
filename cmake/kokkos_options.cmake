########################## NOTES ###############################################
# This files goal is to take CMake options using CMake idioms and map them onto
# the KOKKOS_SETTINGS variables that gets passed to the kokkos makefile
# configuration:
#  make -f ${CMAKE_SOURCE_DIR}/core/src/Makefile ${KOKKOS_SETTINGS} build-makefile-cmake-kokkos
# Which generates KokkosCore_config.h and gen_kokkos.cmake
# To understand how to form KOKKOS_SETTINGS, see
#  <KOKKOS_PATH>/Makefile.kokkos


########################## AVAIALBLE OPTIONS ###################################
# List of possible host architectures.
set(KOKKOS_HOST_ARCH_LIST)
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

# KOKKOS_HOST_ARCH must be defined previoiusly
#set_property(CACHE KOKKOS_HOST_ARCH PROPERTY STRINGS ${KOKKOS_HOST_ARCH_LIST})

# List of possible device architectures.
# The case and spelling here needs to match Makefile.kokkos
set(KOKKOS_DEVICES_LIST)
# Options: Cuda,ROCm,OpenMP,Pthread,Qthreads,Serial
list(APPEND KOKKOS_DEVICES_LIST
    Cuda          # NVIDIA GPU -- see below
    OpenMP        # OpenMP
    Pthread       # pthread
    Qthreads      # qthreads
    Serial        # serial
    ROCm          # Relocatable device code
    )

# List of possible TPLs for Kokkos
# From Makefile.kokkos: Options: hwloc,librt,experimental_memkind
set(KOKKOS_USE_TPLS_LIST)
list(APPEND KOKKOS_USE_TPLS_LIST
    HWLOC          # hwloc
    LIBRT          # librt
    MEMKIND        # experimental_memkind
    )
# Map of cmake variables to Makefile variables
set(KOKKOS_INTERNAL_HWLOC hwloc)
set(KOKKOS_INTERNAL_LIBRT librt)
set(KOKKOS_INTERNAL_MEMKIND experimental_memkind)

# List of possible Options for CUDA
# From Makefile.kokkos: Options: use_ldg,force_uvm,rdc
set(KOKKOS_CUDA_OPTIONS_LIST)
list(APPEND KOKKOS_CUDA_OPTIONS_LIST
    LDG_INTRINSIC              # use_ldg
    UVM                        # force_uvm
    RELOCATABLE_DEVICE_CODE    # rdc
    )
# Map of cmake variables to Makefile variables
set(KOKKOS_INTERNAL_LDG_INTRINSIC use_ldg)
set(KOKKOS_INTERNAL_UVM librt)
set(KOKKOS_INTERNAL_RELOCATABLE_DEVICE_CODE rdc)

######################### INITIALIZE INTERNAL VARIABLES ########################
# As we go along, also set the variables needed for KOKKOS_SETTINGS

# Add Kokkos' modules to CMake's module path.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${Kokkos_SOURCE_DIR}/cmake/Modules/")

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

set(KOKKOS_SEPARATE_LIBS OFF CACHE BOOL "OFF = kokkos.  ON = kokkoscore, kokkoscontainers, and kokkosalgorithms.")

# Enable debugging.
set(KOKKOS_DEBUG OFF CACHE BOOL "Enable debugging in Kokkos.")

# From Makefile.kokkos: Options: yes,no
if(${KOKKOS_DEBUG})
  set(KOKKOS_DEBUG yes)
else()
  set(KOKKOS_DEBUG no)
endif()

# Qthreads options.
set(KOKKOS_QTHREADS_DIR "" CACHE PATH "Location of Qthreads library.")

#------------------------------- KOKKOS_DEVICES --------------------------------
# Set which Kokkos backend to use.
# These are the actual options that define the settings.
set(KOKKOS_ENABLE_CUDA OFF CACHE BOOL "Use Kokkos CUDA backend")
set(KOKKOS_ENABLE_OPENMP ON CACHE BOOL "Use Kokkos OpenMP backend")
set(KOKKOS_ENABLE_PTHREAD OFF CACHE BOOL "Use Kokkos Pthread backend")
set(KOKKOS_ENABLE_QTHREADS OFF CACHE BOOL "Use Kokkos Qthreads backend")
set(KOKKOS_ENABLE_SERIAL ON CACHE BOOL "Use Kokkos Serial backend")

# Can have multiple devices 
set(KOKKOS_DEVICESl)
foreach(devopt ${KOKKOS_DEVICES_LIST})
  string(TOUPPER ${devopt} devoptuc)
  if (${KOKKOS_ENABLE_${devoptuc}}) 
    list(APPEND KOKKOS_DEVICESl ${devopt})
  endif ()
endforeach()
# List needs to be comma-delmitted
string(REPLACE ";" "," KOKKOS_DEVICES "${KOKKOS_DEVICESl}")

#------------------------------- KOKKOS_OPTIONS --------------------------------
# From Makefile.kokkos: Options: aggressive_vectorization,disable_profiling

# Enable aggressive vectorization.
set(KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION OFF CACHE BOOL "Enable aggressive vectorization.")

# Enable profiling.
set(KOKKOS_ENABLE_PROFILING ON CACHE BOOL "Enable profiling.")

set(KOKKOS_OPTIONSl)
if(${KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION})
      list(APPEND KOKKOS_OPTIONSl aggressive_vectorization)
endif()
if(NOT ${KOKKOS_ENABLE_PROFILING})
      list(APPEND KOKKOS_OPTIONSl disable_vectorization)
endif()
# List needs to be comma-delmitted
string(REPLACE ";" "," KOKKOS_OPTIONS "${KOKKOS_OPTIONSl}")


#------------------------------- KOKKOS_USE_TPLS -------------------------------
# Enable hwloc library.
set(KOKKOS_ENABLE_HWLOC OFF CACHE BOOL "Enable hwloc for better process placement.")
set(KOKKOS_HWLOC_DIR "" CACHE PATH "Location of hwloc library.")

# Enable memkind library.
set(KOKKOS_ENABLE_MEMKIND OFF CACHE BOOL "Enable memkind.")
set(KOKKOS_MEMKIND_DIR "" CACHE PATH "Location of memkind library.")

set(KOKKOS_ENABLE_LIBRT OFF CACHE BOOL "Enable librt for more precise timer.")

# Construct the Makefile options
set(KOKKOS_USE_TPLSl)
foreach(tplopt ${KOKKOS_USE_TPLS_LIST})
  if (${KOKKOS_ENABLE_${tplopt}}) 
    list(APPEND KOKKOS_USE_TPLSl ${KOKKOS_INTERNAL_${tplopt}})
  endif ()
endforeach()
# List needs to be comma-delmitted
string(REPLACE ";" "," KOKKOS_USE_TPLS "${KOKKOS_USE_TPLSl}")


#------------------------------- KOKKOS_CUDA_OPTIONS ---------------------------
# CUDA options.
set(KOKKOS_CUDA_DIR "" CACHE PATH "Location of CUDA library.  Defaults to where nvcc installed.")
set(KOKKOS_ENABLE_CUDA_LDG_INTRINSIC OFF CACHE BOOL "Enable CUDA LDG.")
set(KOKKOS_ENABLE_CUDA_UVM OFF CACHE BOOL "Enable CUDA unified virtual memory.")
set(KOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE OFF CACHE BOOL "Enable relocatable device code for CUDA.")
#SEK: Deprecated?
set(KOKKOS_ENABLE_CUDA_LAMBDA ON CACHE BOOL "Enable lambdas for CUDA.")

# Construct the Makefile options
set(KOKKOS_CUDA_OPTIONS)
foreach(cudaopt ${KOKKOS_CUDA_OPTIONS_LIST})
  if (${KOKKOS_ENABLE_${cudaopt}}) 
    list(APPEND KOKKOS_CUDA_OPTIONSl ${KOKKOS_INTERNAL_${cudaopt}})
  endif ()
endforeach()
# List needs to be comma-delmitted
string(REPLACE ";" "," KOKKOS_CUDA_OPTIONS "${KOKKOS_CUDA_OPTIONSl}")

#------------------------------- PATH VARIABLES --------------------------------
#  Want makefile to use same executables specified which means modifying
#  the path so the $(shell ...) commands in the makefile see the right exec
#  Also, the Makefile's use FOO_PATH naming scheme for -I/-L construction
#TODO:  Makefile.kokkos allows this to be overwritten? ROCM_HCC_PATH

set(KOKKOS_INTERNAL_PATHS)
set(addpathl)
foreach(kvar "CUDA;QTHREADS;${KOKKOS_USE_TPLS_LIST}")
  if(${KOKKOS_ENABLE_${kvar}})
    if(DEFINED KOKKOS_${kvar}_DIR)
      set(KOKKOS_INTERNAL_PATHS "${KOKKOS_INTERNAL_PATHS} ${kvar}_PATH=${KOKKOS_${kvar}_DIR}")
      if(IS_DIRECTORY ${KOKKOS_${kvar}_DIR}/bin)
        list(APPEND addpathl ${KOKKOS_${kvar}_DIR}/bin)
      endif()
    endif()
  endif()
endforeach()
# Path env is : delimitted
string(REPLACE ";" ":" KOKKOS_INTERNAL_ADDTOPATH "${addpathl}")

######################### SET KOKKOS_SETTINGS ##################################
# Set the KOKKOS_SETTINGS String -- this is the primary communication with the
# makefile configuration.  See Makefile.kokkos

set(KOKKOS_SETTINGS KOKKOS_SRC_PATH=${KOKKOS_SRC_PATH})
set(KOKKOS_SETTINGS ${KOKKOS_SETTINGS} KOKKOS_PATH=${KOKKOS_PATH})

# Form of KOKKOS_foo=$KOKKOS_foo
foreach(kvar ARCH;DEVICES;DEBUG;OPTIONS;CUDA_OPTIONS;USE_TPLS)
  set(KOKKOS_VAR KOKKOS_${kvar})
  if(DEFINED KOKKOS_${kvar})
    if (NOT "${${KOKKOS_VAR}}" STREQUAL "")
      set(KOKKOS_SETTINGS ${KOKKOS_SETTINGS} ${KOKKOS_VAR}=${${KOKKOS_VAR}})
    endif()
  endif()
endforeach()

# Form of VAR=VAL
#TODO:  Makefile supports MPICH_CXX, OMPI_CXX as well
foreach(ovar CXX;CXXFLAGS;LDFLAGS)
  if(DEFINED ${ovar})
    if (NOT "${${ovar}}" STREQUAL "")
      set(KOKKOS_SETTINGS ${KOKKOS_SETTINGS} ${ovar}=${${ovar}})
    endif()
  endif()
endforeach()

# Finally, do the paths
if (NOT "${KOKKOS_INTERNAL_PATHS}" STREQUAL "")
  set(KOKKOS_SETTINGS ${KOKKOS_SETTINGS} ${KOKKOS_INTERNAL_PATHS})
endif()
if (NOT "${KOKKOS_INTERNAL_ADDTOPATH}" STREQUAL "")
  set(KOKKOS_SETTINGS ${KOKKOS_SETTINGS} PATH=${KOKKOS_INTERNAL_ADDTOPATH}:\${PATH})
endif()

# Final form that gets passed to make
set(KOKKOS_SETTINGS env ${KOKKOS_SETTINGS})

############################ PRINT CONFIGURE STATUS ############################

if(KOKKOS_CMAKE_VERBOSE)
  message(STATUS "")
  message(STATUS "****************** Kokkos Settings ******************")
  message(STATUS "Execution Spaces")

  if(KOKKOS_ENABLE_CUDA)
    message(STATUS "  Device Parallel: Cuda")
  else()
    message(STATUS "  Device Parallel: None")
  endif()

  if(KOKKOS_ENABLE_OPENMP)
    message(STATUS "    Host Parallel: OpenMP")
  elseif(KOKKOS_ENABLE_PTHREAD)
    message(STATUS "    Host Parallel: Pthread")
  elseif(KOKKOS_ENABLE_QTHREADS)
    message(STATUS "    Host Parallel: Qthreads")
  else()
    message(STATUS "    Host Parallel: None")
  endif()

  if(KOKKOS_ENABLE_SERIAL)
    message(STATUS "      Host Serial: Serial")
  else()
    message(STATUS "      Host Serial: None")
  endif()

  message(STATUS "")
  message(STATUS "Architectures")
  message(STATUS "    Host Architecture: ${KOKKOS_HOST_ARCH}")
  message(STATUS "  Device Architecture: ${KOKKOS_GPU_ARCH}")

  message(STATUS "")
  message(STATUS "Enabled options")

  if(KOKKOS_SEPARATE_LIBS)
    message(STATUS "  KOKKOS_SEPARATE_LIBS")
  endif()

  if(KOKKOS_ENABLE_HWLOC)
    message(STATUS "  KOKKOS_ENABLE_HWLOC")
  endif()

  if(KOKKOS_ENABLE_MEMKIND)
    message(STATUS "  KOKKOS_ENABLE_MEMKIND")
  endif()

  if(KOKKOS_DEBUG)
    message(STATUS "  KOKKOS_DEBUG")
  endif()

  if(KOKKOS_ENABLE_PROFILING)
    message(STATUS "  KOKKOS_ENABLE_PROFILING")
  endif()

  if(KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION)
    message(STATUS "  KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION")
  endif()

  if(KOKKOS_ENABLE_CUDA)
    if(KOKKOS_ENABLE_CUDA_LDG_INTRINSIC)
      message(STATUS "  KOKKOS_ENABLE_CUDA_LDG_INTRINSIC")
    endif()

    if(KOKKOS_ENABLE_CUDA_UVM)
      message(STATUS "  KOKKOS_ENABLE_CUDA_UVM")
    endif()

    if(KOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE)
      message(STATUS "  KOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE")
    endif()

    if(KOKKOS_ENABLE_CUDA_LAMBDA)
      message(STATUS "  KOKKOS_ENABLE_CUDA_LAMBDA")
    endif()

    if(KOKKOS_CUDA_DIR)
      message(STATUS "  KOKKOS_CUDA_DIR: ${KOKKOS_CUDA_DIR}")
    endif()
  endif()

  if(KOKKOS_QTHREADS_DIR)
    message(STATUS "  KOKKOS_QTHREADS_DIR: ${KOKKOS_QTHREADS_DIR}")
  endif()

  if(KOKKOS_HWLOC_DIR)
    message(STATUS "  KOKKOS_HWLOC_DIR: ${KOKKOS_HWLOC_DIR}")
  endif()

  if(KOKKOS_MEMKIND_DIR)
    message(STATUS "  KOKKOS_MEMKIND_DIR: ${KOKKOS_MEMKIND_DIR}")
  endif()

  message(STATUS "")
  message(STATUS "Final kokkos settings variable:")
  message(STATUS "  ${KOKKOS_SETTINGS}")

  message(STATUS "*****************************************************")
  message(STATUS "")
endif()
