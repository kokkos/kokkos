########################## NOTES ###############################################
#  List the options for configuring kokkos using CMake method of doing it.
#  These options then get mapped onto KOKKOS_SETTINGS environment variable by
#  kokkos_settings.cmake.  It is separate to allow other packages to override
#  these variables (e.g., TriBITS).

########################## AVAIALBLE OPTIONS ###################################
# Use lists for documentation, verification, and programming convenience

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
string(REPLACE ";" ${tmpr} KOKKOS_INTERNAL_ARCH_DOCSTR "${KOKKOS_HOST_ARCH_LIST}")
# This would be useful, but we use Foo_ENABLE mechanisms
#string(REPLACE ";" ${tmpr} KOKKOS_INTERNAL_DEVICES_DOCSTR "${KOKKOS_DEVICES_LIST}")
#string(REPLACE ";" ${tmpr} KOKKOS_INTERNAL_USE_TPLS_DOCSTR "${KOKKOS_USE_TPLS_LIST}")
#string(REPLACE ";" ${tmpr} KOKKOS_INTERNAL_CUDA_OPTIONS_DOCSTR "${KOKKOS_CUDA_OPTIONS_LIST}")

#------------------------------- GENERAL OPTIONS -------------------------------
# KOKKOS_HOST_ARCH must be defined previoiusly
#set_property(CACHE KOKKOS_HOST_ARCH PROPERTY STRINGS ${KOKKOS_HOST_ARCH_LIST})

# Setting this variable to a value other than "None" can improve host
# performance by turning on architecture specific code.
# TODO:  Documentation should show the entire options
set(KOKKOS_HOST_ARCH "None" CACHE STRING 
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
set(KOKKOS_ENABLE_CUDA OFF CACHE BOOL "Use Kokkos CUDA backend")
set(KOKKOS_ENABLE_OPENMP ON CACHE BOOL "Use Kokkos OpenMP backend")
set(KOKKOS_ENABLE_PTHREAD OFF CACHE BOOL "Use Kokkos Pthread backend")
set(KOKKOS_ENABLE_QTHREADS OFF CACHE BOOL "Use Kokkos Qthreads backend")
set(KOKKOS_ENABLE_SERIAL ON CACHE BOOL "Use Kokkos Serial backend")


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


#------------------------------- DEPRECATED OPTIONS  ---------------------------
set(Kokkos_ENABLE_Debug_Bounds_Check OFF CACHE BOOL "Deprecated -- has no effect")
set(KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK OFF CACHE BOOL "Deprecated -- has no effect")
set(KOKKOS_ENABLE_DEBUG OFF CACHE BOOL "Deprecated -- has no effect")
set(Kokkos_ENABLE_Winthread OFF CACHE BOOL "Deprecated -- has no effect")
set(Kokkos_USING_DEPRECATED_VIEW OFF CACHE BOOL "Deprecated -- has no effect")
