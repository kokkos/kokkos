########################## NOTES ###############################################
#  List the options for configuring kokkos using CMake method of doing it.
#  These options then get mapped onto KOKKOS_SETTINGS environment variable by
#  kokkos_settings.cmake.  It is separate to allow other packages to override
#  these variables (e.g., TriBITS).

########################## AVAILABLE OPTIONS ###################################
# Use lists for documentation, verification, and programming convenience

# List of possible host architectures.
set(KOKKOS_ARCH_LIST)
list(APPEND KOKKOS_ARCH_LIST
     None            # No architecture optimization
     AMDAVX          # (HOST) AMD chip
     ARMv80          # (HOST) ARMv8.0 Compatible CPU
     ARMv81          # (HOST) ARMv8.1 Compatible CPU
     ARMv8-ThunderX  # (HOST) ARMv8 Cavium ThunderX CPU
     SNB             # (HOST) Intel Sandy/Ivy Bridge CPUs
     HSW             # (HOST) Intel Haswell CPUs
     BDW             # (HOST) Intel Broadwell Xeon E-class CPUs
     SKX             # (HOST) Intel Sky Lake Xeon E-class HPC CPUs (AVX512)
     KNC             # (HOST) Intel Knights Corner Xeon Phi
     KNL             # (HOST) Intel Knights Landing Xeon Phi
     BGQ             # (HOST) IBM Blue Gene Q
     Power7          # (HOST) IBM POWER7 CPUs
     Power8          # (HOST) IBM POWER8 CPUs
     Power9          # (HOST) IBM POWER9 CPUs
     Kepler          # (GPU) NVIDIA Kepler default (generation CC 3.5)
     Kepler30        # (GPU) NVIDIA Kepler generation CC 3.0
     Kepler32        # (GPU) NVIDIA Kepler generation CC 3.2
     Kepler35        # (GPU) NVIDIA Kepler generation CC 3.5
     Kepler37        # (GPU) NVIDIA Kepler generation CC 3.7
     Maxwell         # (GPU) NVIDIA Maxwell default (generation CC 5.0)
     Maxwell50       # (GPU) NVIDIA Maxwell generation CC 5.0
     Maxwell52       # (GPU) NVIDIA Maxwell generation CC 5.2
     Maxwell53       # (GPU) NVIDIA Maxwell generation CC 5.3
     Pascal60        # (GPU) NVIDIA Pascal generation CC 6.0
     Pascal61        # (GPU) NVIDIA Pascal generation CC 6.1
    )

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

# List of possible Advanced options
set(KOKKOS_OPTIONS_LIST)
list(APPEND KOKKOS_OPTIONS_LIST
       AGGRESSIVE_VECTORIZATION    
       DISABLE_PROFILING          
       DISABLE_DUALVIEW_MODIFY_CHECK
       ENABLE_PROFILE_LOAD_PRINT   
    )
# Map of cmake variables to Makefile variables
set(KOKKOS_INTERNAL_LDG_INTRINSIC use_ldg)
set(KOKKOS_INTERNAL_UVM librt)
set(KOKKOS_INTERNAL_RELOCATABLE_DEVICE_CODE rdc)


# List of possible Options for CUDA
# From Makefile.kokkos: Options: use_ldg,force_uvm,rdc
set(KOKKOS_CUDA_OPTIONS_LIST)
list(APPEND KOKKOS_CUDA_OPTIONS_LIST
    LDG_INTRINSIC              # use_ldg
    UVM                        # force_uvm
    RELOCATABLE_DEVICE_CODE    # rdc
    LAMBDA                     # lambda
    )
# Map of cmake variables to Makefile variables
set(KOKKOS_INTERNAL_LDG_INTRINSIC use_ldg)
set(KOKKOS_INTERNAL_UVM force_uvm)
set(KOKKOS_INTERNAL_RELOCATABLE_DEVICE_CODE rdc)
set(KOKKOS_INTERNAL_LAMBDA lambda)


#------------------------------- Create doc strings ----------------------------
set(tmpr "\n       ")
string(REPLACE ";" ${tmpr} KOKKOS_INTERNAL_ARCH_DOCSTR "${KOKKOS_ARCH_LIST}")
# This would be useful, but we use Foo_ENABLE mechanisms
#string(REPLACE ";" ${tmpr} KOKKOS_INTERNAL_DEVICES_DOCSTR "${KOKKOS_DEVICES_LIST}")
#string(REPLACE ";" ${tmpr} KOKKOS_INTERNAL_USE_TPLS_DOCSTR "${KOKKOS_USE_TPLS_LIST}")
#string(REPLACE ";" ${tmpr} KOKKOS_INTERNAL_CUDA_OPTIONS_DOCSTR "${KOKKOS_CUDA_OPTIONS_LIST}")

#------------------------------- GENERAL OPTIONS -------------------------------
# KOKKOS_ARCH must be defined previously
#set_property(CACHE KOKKOS_ARCH PROPERTY STRINGS ${KOKKOS_ARCH_LIST})

# Setting this variable to a value other than "None" can improve host
# performance by turning on architecture specific code.
# NOT SET is used to determine if the option is passed in.  It is reset to
# default "None" down below.
set(KOKKOS_ARCH "NOT_SET" CACHE STRING 
      "Optimize for specific host architecture. Options are: ${KOKKOS_INTERNAL_ARCH_DOCSTR}")

# Whether to build separate libraries or now
set(KOKKOS_SEPARATE_LIBS OFF CACHE BOOL "OFF = kokkos.  ON = kokkoscore, kokkoscontainers, and kokkosalgorithms.")

# Enable debugging.
set(KOKKOS_DEBUG OFF CACHE BOOL "Enable debugging in Kokkos.")

# Qthreads options.
set(KOKKOS_QTHREADS_DIR "" CACHE PATH "Location of Qthreads library.")


#------------------------------- KOKKOS_DEVICES --------------------------------
# Set which Kokkos backend to use.
# These are the actual options that define the settings.
set(KOKKOS_ENABLE_CUDA OFF CACHE BOOL "Enable CUDA support in Kokkos.")
set(KOKKOS_ENABLE_OPENMP OFF CACHE BOOL "Enable OpenMP support in Kokkos.")
set(KOKKOS_ENABLE_PTHREAD OFF CACHE BOOL "Enable Pthread support in Kokkos.")
set(KOKKOS_ENABLE_QTHREADS OFF CACHE BOOL "Enable Qthreads support in Kokkos.")
set(KOKKOS_ENABLE_SERIAL ON CACHE BOOL "Whether to enable the Kokkos::Serial device.  This device executes \"parallel\" kernels sequentially on a single CPU thread.  It is enabled by default.  If you disable this device, please enable at least one other CPU device, such as Kokkos::OpenMP or Kokkos::Threads.")

#------------------------------- KOKKOS_OPTIONS --------------------------------
# From Makefile.kokkos: Advanced Options: 
#compiler_warnings, aggressive_vectorization, disable_profiling, disable_dualview_modify_check, enable_profile_load_print
# TriBITS had mixed case so adding both versions for backwards compatibility

# Enable compiler warnings
set(KOKKOS_ENABLE_COMPILER_WARNINGS OFF CACHE BOOL "Enable compiler warnings.")

# Enable aggressive vectorization.
set(KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION OFF CACHE BOOL "Enable aggressive vectorization.")

# Enable profiling.
set(KOKKOS_ENABLE_PROFILING ON CACHE BOOL "Enable profiling.")
set(KOKKOS_ENABLE_Profiling ${KOKKOS_ENABLE_Profiling})  

set(KOKKOS_ENABLE_DUALVIEW_MODIFY_CHECK ON CACHE BOOL "Enable dualview modify check.")
set(Kokkos_ENABLE_Debug_DualView_Modify_Check ${KOKKOS_ENABLE_DUALVIEW_MODIFY_CHECK})


set(KOKKOS_ENABLE_PROFILE_LOAD_PRINT OFF CACHE BOOL "Enable profile load print.")
set(Kokkos_ENABLE_Profiling_Load_Print ${KOKKOS_ENABLE_PROFILE_LOAD_PRINT})

#------------------------------- KOKKOS_USE_TPLS -------------------------------
# Enable hwloc library.
set(KOKKOS_ENABLE_HWLOC OFF CACHE BOOL "Enable hwloc for better process placement.")
set(KOKKOS_HWLOC_DIR "" CACHE PATH "Location of hwloc library. (kokkos tpl)")

# Enable memkind library.
set(KOKKOS_ENABLE_MEMKIND OFF CACHE BOOL "Enable memkind. (kokkos tpl)")
set(KOKKOS_MEMKIND_DIR "" CACHE PATH "Location of memkind library. (kokkos tpl)")

set(KOKKOS_ENABLE_LIBRT OFF CACHE BOOL "Enable librt for more precise timer.  (kokkos tpl)")


#------------------------------- KOKKOS_CUDA_OPTIONS ---------------------------
# CUDA options.
set(KOKKOS_CUDA_DIR "" CACHE PATH "Location of CUDA library.  Defaults to where nvcc installed.")
set(KOKKOS_ENABLE_CUDA_LDG_INTRINSIC OFF CACHE BOOL "Enable CUDA LDG. (cuda option)") 
set(KOKKOS_ENABLE_CUDA_UVM OFF CACHE BOOL "Enable CUDA unified virtual memory.")
set(KOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE OFF CACHE BOOL "Enable relocatable device code for CUDA. (cuda option)")
#SEK: Deprecated?
set(KOKKOS_ENABLE_CUDA_LAMBDA ON CACHE BOOL "Enable lambdas for CUDA. (cuda option)")


#----------------------- HOST ARCH AND LEGACY TRIBITS --------------------------
# This defines the previous legacy TriBITS builds. 
set(KOKKOS_LEGACY_TRIBITS False)
IF ("${KOKKOS_ARCH}" STREQUAL "NOT_SET")
  set(KOKKOS_ARCH "None")
  IF(KOKKOS_HAS_TRILINOS)
    set(KOKKOS_LEGACY_TRIBITS True)
  ENDIF()
ENDIF()
IF (KOKKOS_HAS_TRILINOS)
  IF (KOKKOS_LEGACY_TRIBITS)
    message(STATUS "Using the legacy tribits build because KOKKOS_ARCH not set")
  ELSE()
    message(STATUS "NOT using the legacy tribits build because KOKKOS_ARCH *is* set")
  ENDIF()
ENDIF()


#------------------------------- DEPRECATED OPTIONS  ---------------------------
# Previous TriBITS builds used TRIBITS_ADD_OPTION_AND_DEFINE extensively
# Because the options have moved to this file, we need to add the defines
# Also TriBITS used a different capitalization so handle that.
if(KOKKOS_LEGACY_TRIBITS)
  set(Kokkos_ENABLE_DEBUG OFF CACHE BOOL "Deprecated -- Please use KOKKOS_DEBUG")
  IF (Kokkos_ENABLE_DEBUG)
    set(KOKKOS_DEBUG True)                  # New Option
    set(KOKKOS_HAVE_DEBUG True)             # Define
  ENDIF ()

  set(Kokkos_ENABLE_Serial OFF CACHE BOOL "Deprecated -- Please use KOKKOS_ENABLE_SERIAL")
  IF (Kokkos_ENABLE_Serial)
    set(KOKKOS_ENABLE_SERIAL True)          # New Option
    set(KOKKOS_HAVE_SERIAL True)            # Define
  ENDIF ()

  set(Kokkos_ENABLE_Pthread OFF CACHE BOOL "Deprecated -- Please use KOKKOS_ENABLE_PTHREAD")
  IF (Kokkos_ENABLE_Pthread)
    set(KOKKOS_ENABLE_PTHREAD True)          # New Option
    set(KOKKOS_HAVE_PTHREAD True)            # Define
    ADD_DEFINITIONS(-DGTEST_HAS_PTHREAD=0)
  ENDIF ()

  set(Kokkos_ENABLE_OpenMP OFF CACHE BOOL "Deprecated -- Please use KOKKOS_ENABLE_OPENMP")
  IF (Kokkos_ENABLE_OpenMP)
    set(KOKKOS_ENABLE_OPENMP True)          # New Option
    set(KOKKOS_HAVE_OPENMP True)            # Define
  ENDIF ()

  set(Kokkos_ENABLE_QTHREAD OFF CACHE BOOL "Deprecated -- Please use KOKKOS_ENABLE_QTHREADS")
  IF (Kokkos_ENABLE_QTHREAD)
    set(KOKKOS_ENABLE_QTHREADS True)          # New Option
    set(KOKKOS_HAVE_QTHREAD True)             # Define
  ENDIF ()

  set(Kokkos_ENABLE_Cuda OFF CACHE BOOL "Deprecated -- Please use KOKKOS_ENABLE_CUDA")
  IF (Kokkos_ENABLE_Cuda)
    set(KOKKOS_ENABLE_CUDA True)           # New Option
    set(KOKKOS_HAVE_CUDA True)             # Define
  ENDIF ()

  set(Kokkos_ENABLE_Cuda_UVM OFF CACHE BOOL "Deprecated -- Please use KOKKOS_ENABLE_CUDA_UVM")
  IF (Kokkos_ENABLE_Cuda_UVM)
    set(KOKKOS_ENABLE_CUDA_UVM True)       # New Option
    set(KOKKOS_USE_CUDA_UVM True)          # Define
  ENDIF ()

  set(Kokkos_ENABLE_Cuda_RDC OFF CACHE BOOL "Deprecated -- Please use KOKKOS_ENABLE_CUDA_RDC")
  IF (Kokkos_ENABLE_Cuda_RDC)
    set(KOKKOS_ENABLE_CUDA_RDC True)       # New Option
    set(KOKKOS_HAVE_CUDA_RDC True)         # Define
  ENDIF ()

  set(Kokkos_ENABLE_Cuda_Lambda OFF CACHE BOOL "Deprecated -- Please use KOKKOS_ENABLE_CUDA_LAMBDA")
  IF (Kokkos_ENABLE_Cuda_Lambda)
    set(KOKKOS_ENABLE_CUDA_LAMBDA True)       # New Option
    set(KOKKOS_HAVE_CUDA_LAMBDA True)         # Define
  ENDIF ()


  set(Kokkos_ENABLE_HWLOC OFF CACHE BOOL "Deprecated -- Please use KOKKOS_ENABLE_HWLOC")
  IF (Kokkos_ENABLE_HWLOC)
    set(KOKKOS_ENABLE_HWLOC True)       # New Option
    set(KOKKOS_HAVE_HWLOC True)         # Define
  ENDIF ()

  set(Kokkos_ENABLE_Debug_Bounds_Check OFF CACHE BOOL "Deprecated -- has no effect")
  set(KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK OFF CACHE BOOL "Deprecated -- has no effect")

  set(Kokkos_ENABLE_Winthread OFF CACHE BOOL "Deprecated -- has no effect")
  set(Kokkos_USING_DEPRECATED_VIEW OFF CACHE BOOL "Deprecated -- has no effect")
  set(Kokkos_ENABLE_CXX11 OFF CACHE BOOL "Deprecated -- has no effect")
ELSE()
  IF (KOKKOS_ENABLE_SERIAL)
    set(Kokkos_ENABLE_Serial True)
  endif()
  IF (KOKKOS_ENABLE_OPENMP)
    set(Kokkos_ENABLE_OpenMP True)
  endif()
  IF (KOKKOS_ENABLE_PTHREAD)
    set(Kokkos_ENABLE_Pthread True)
  endif()
  IF (KOKKOS_ENABLE_CUDA)
    set(Kokkos_ENABLE_Cuda True)
  endif()
ENDIF()


#------------------------------- Mapping of Trilinos options -------------------
# Map tribits settings onto kokkos settings 
IF(Trilinos_ENABLE_Kokkos)
  set(KOKKOS_ENABLE_PTHREAD ${TPL_ENABLE_Pthread})
  set(KOKKOS_ENABLE_QTHREADS ${TPL_ENABLE_QTHREAD})
  set(KOKKOS_ENABLE_OPENMP ${Trilinos_ENABLE_OpenMP})
  # No tribits equivalent
  #set(KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION ${TPL_ENABLE_})
  if (${TPL_ENABLE_MPI})
    set(KOKKOS_ENABLE_SERIAL OFF)
  else()
    set(KOKKOS_ENABLE_SERIAL ON)
  endif()

  # Handle Kokkos TPLs
  set(KOKKOS_ENABLE_HWLOC ${TPL_ENABLE_HWLOC})
  # Worry about later
  #set(KOKKOS_HWLOC_DIR ${TPL_ENABLE_HWLOC})

  #TODO
  # Enable memkind library.
  #set(KOKKOS_ENABLE_MEMKIND ${TPL_ENABLE_MEMKIND})
  # Worry about later
  #set(KOKKOS_MEMKIND_DIR 

  # Not done in TriBITS
  #set(KOKKOS_ENABLE_LIBRT ${TPL_ENABLE_LIBRT})
ENDIF ()
