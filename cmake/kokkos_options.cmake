########################## NOTES ###############################################
#  List the options for configuring kokkos using CMake method of doing it.
#  These options then get mapped onto KOKKOS_SETTINGS environment variable by
#  kokkos_settings.cmake.  It is separate to allow other packages to override
#  these variables (e.g., TriBITS).

########################## AVAILABLE OPTIONS ###################################
# Use lists for documentation, verification, and programming convenience

function(KOKKOS_ENABLE_OPTION CAMEL_SUFFIX DEFAULT DOCSTRING)
  set(CAMEL_NAME Kokkos_ENABLE_${CAMEL_SUFFIX})
  string(TOUPPER ${CAMEL_NAME} UC_NAME)
  if (NOT DEFINED KOKKOS_CACHED_${UC_NAME} AND DEFINED ${CAMEL_NAME})
    #this is our first time through the cmake
    #we were given the camel case name instead of the UC name we wanted
    #make darn sure we don't have both an UC and Camel version that differ
    if (DEFINED ${UC_NAME} AND NOT ${CAMEL_NAME} STREQUAL ${UC_NAME})
      message(FATAL_ERROR "Given both ${CAMEL_NAME} and ${UC_NAME} with different values")
    endif()
    #great, no conflicts - use the camel case name as the default for the UC
    set(${UC_NAME} ${${CAMEL_NAME}} CACHE BOOL ${DOCSTRING})
  elseif(DEFINED ${CAMEL_NAME})
    #this is at least our second configure and we have an existing cache
    #CMake makes this impossible to distinguish something already in cache
    #and somthing given explicitly on the command line
    #at this point, we have no choice but to accept the Camel value and print a warning
    if (NOT ${CAMEL_NAME} STREQUAL ${UC_NAME})
      message(WARNING "Overriding ${UC_NAME}=${${UC_NAME}} with ${CAMEL_NAME}=${${CAMEL_NAME}}")
    endif()
    #I have to accept the Camel case value - really no choice here - force it
    set(${UC_NAME} ${${CAMEL_NAME}} CACHE BOOL ${DOCSTRING} FORCE)
  else() #great, no camel case names - nice and simple
    set(${UC_NAME} ${DEFAULT} CACHE BOOL ${DOCSTRING})
  endif()
  set(KOKKO_CACHED_${UC_NAME} ${${UC_NAME}} CACHE INTERNAL ${DOCSTRING})

  if (${UC_NAME}) #cmake if statements follow really annoying string resolution rules
    message(STATUS "${UC_NAME}") 
  endif()
endfunction()

KOKKOS_ENABLE_OPTION(TESTS         OFF  "Whether to build serial  backend")
KOKKOS_ENABLE_OPTION(EXAMPLES      OFF  "Whether to build OpenMP  backend")
KOKKOS_ENABLE_OPTION(Serial        ON  "Whether to build serial  backend")
KOKKOS_ENABLE_OPTION(OpenMP        OFF "Whether to build OpenMP  backend")
KOKKOS_ENABLE_OPTION(Pthread       OFF "Whether to build Pthread backend")
KOKKOS_ENABLE_OPTION(Cuda          OFF "Whether to build CUDA backend")
KOKKOS_ENABLE_OPTION(ROCm          OFF "Whether to build AMD ROCm backend")
KOKKOS_ENABLE_OPTION(HWLOC         OFF "Whether to enable HWLOC features - may also require -DHWLOC_DIR")
KOKKOS_ENABLE_OPTION(MEMKIND       OFF "Whether to enable MEMKIND featuers - may also require -DMEMKIND_DIR")
KOKKOS_ENABLE_OPTION(LIBRT         OFF "Whether to enable LIBRT features")
KOKKOS_ENABLE_OPTION(Cuda_Relocatable_Device_Code  OFF "Whether to enable relocatable device code (RDC) for CUDA")
KOKKOS_ENABLE_OPTION(Cuda_UVM             OFF "Whether to enable unified virtual memory (UVM) for CUDA")
KOKKOS_ENABLE_OPTION(Cuda_LDG_Intrinsic   OFF "Whether to use CUDA LDG intrinsics")
KOKKOS_ENABLE_OPTION(HPX_ASYNC_DISPATCH   OFF "Whether HPX supports asynchronous dispath")
KOKKOS_ENABLE_OPTION(Debug                OFF "Whether to activate extra debug features - may increase compile times")
KOKKOS_ENABLE_OPTION(Debug_DualView_Modify_Check OFF "Debug check on dual views")
KOKKOS_ENABLE_OPTION(Debug_Bounds_Check   OFF "Whether to use bounds checking - will increase runtime") 
KOKKOS_ENABLE_OPTION(Compiler_Warnings    OFF "Whether to print all compiler warnings")
KOKKOS_ENABLE_OPTION(Profiling            ON  "Whether to create bindings for profiling tools")
KOKKOS_ENABLE_OPTION(Profiling_Load_Print OFF "Whether to print information about which profiling tools got loaded")
KOKKOS_ENABLE_OPTION(Aggressive_Vectorization OFF "Whether to aggressively vectorize loops")
KOKKOS_ENABLE_OPTION(Deprecated_Code      OFF "Whether to enable deprecated code")
KOKKOS_ENABLE_OPTION(Explicit_Instantiation OFF 
  "Whether to explicitly instantiate certain types to lower future compile times")
SET(KOKKOS_ENABLE_ETI ${KOKKOS_ENABLE_EXPLICIT_INSTANTIATION} CACHE INTERNAL "eti")

IF(Trilinos_ENABLE_Kokkos AND TPL_ENABLE_QTHREAD)             
SET(QTHR_DEFAULT ON)
ELSE()
SET(QTHR_DEFAULT OFF)
ENDIF()
KOKKOS_ENABLE_OPTION(Qthread ${QTHR_DEFAULT} 
  "Whether to build Qthreads backend - may also require -DQTHREAD_DIR")

IF(Trilinos_ENABLE_Kokkos AND TPL_ENABLE_HPX)
SET(HPX_DEFAULT ON)
ELSE()
SET(HPX_DEFAULT OFF)
ENDIF()
KOKKOS_ENABLE_OPTION(HPX ${HPX_DEFAULT} "Whether to build HPX backend - may also require -DHPX_DIR")

IF(Trilinos_ENABLE_Kokkos AND Trilinos_ENABLE_OpenMP)
  SET(OMP_DEFAULT ON)
ELSE()
  SET(OMP_DEFAULT OFF)
ENDIF()
KOKKOS_ENABLE_OPTION(OpenMP ${OMP_DEFAULT} "Whether to build OpenMP backend")

IF(Trilinos_ENABLE_Kokkos AND TPL_ENABLE_CUDA)
  SET(CUDA_DEFAULT ON)
ELSE()
  SET(CUDA_DEFAULT OFF)
ENDIF()
KOKKOS_ENABLE_OPTION(Cuda ${CUDA_DEFAULT} "Whether to build CUDA backend")

IF (DEFINED CUDA_VERSION AND CUDA_VERSION VERSION_GREATER "7.0")
  SET(LAMBDA_DEFAULT ON)
ELSE()
  SET(LAMBDA_DEFAULT OFF)
ENDIF()
KOKKOS_ENABLE_OPTION(Cuda_Lambda ${LAMBDA_DEFAULT} "Whether to activate experimental laambda features")

IF(DEFINED Kokkos_ARCH)
  MESSAGE(FATAL_ERROR "Defined Kokkos_ARCH, use KOKKOS_ARCH instead!")
ENDIF()
IF(DEFINED Kokkos_Arch)
  MESSAGE(FATAL_ERROR "Defined Kokkos_Arch, use KOKKOS_ARCH instead!")
ENDIF()
  
#-------------------------------------------------------------------------------
# List of possible host architectures.
#-------------------------------------------------------------------------------
set(KOKKOS_ARCH_LIST)
list(APPEND KOKKOS_ARCH_LIST
     None            # No architecture optimization
     AMDAVX          # (HOST) AMD chip
     ARMv80          # (HOST) ARMv8.0 Compatible CPU
     ARMv81          # (HOST) ARMv8.1 Compatible CPU
     ARMv8_ThunderX  # (HOST) ARMv8 Cavium ThunderX CPU
     ARMv8_TX2       # (HOST) ARMv8 Cavium ThunderX2 CPU
     WSM             # (HOST) Intel Westmere CPU
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
     Volta70         # (GPU) NVIDIA Volta generation CC 7.0
     Volta72         # (GPU) NVIDIA Volta generation CC 7.2
     Turing75         # (GPU) NVIDIA Turing generation CC 7.5
     Ryzen
     Epyc
     Kaveri
     Carrizo
     Fiji
     Vega
     GFX901
    )


FOREACH(Arch ${KOKKOS_ARCH_LIST})
  STRING(TOUPPER ${Arch} ARCH)
  SET(KOKKOS_ARCH_${ARCH} OFF CACHE BOOL "Whether to optimize for the ${ARCH} architecture")
ENDFOREACH()

set(tmpr "\n       ")
string(REPLACE ";" ${tmpr} KOKKOS_INTERNAL_ARCH_DOCSTR "${KOKKOS_ARCH_LIST}")
set(KOKKOS_INTERNAL_ARCH_DOCSTR "${tmpr}${KOKKOS_INTERNAL_ARCH_DOCSTR}")
# This would be useful, but we use Foo_ENABLE mechanisms
#string(REPLACE ";" ${tmpr} KOKKOS_INTERNAL_DEVICES_DOCSTR "${KOKKOS_DEVICES_LIST}")
#string(REPLACE ";" ${tmpr} KOKKOS_INTERNAL_USE_TPLS_DOCSTR "${KOKKOS_USE_TPLS_LIST}")
#string(REPLACE ";" ${tmpr} KOKKOS_INTERNAL_CUDA_OPTIONS_DOCSTR "${KOKKOS_CUDA_OPTIONS_LIST}")

#-------------------------------------------------------------------------------
#------------------------------- GENERAL OPTIONS -------------------------------
#-------------------------------------------------------------------------------

# Setting this variable to a value other than "None" can improve host
# performance by turning on architecture specific code.
# NOT SET is used to determine if the option is passed in.  It is reset to
# default "None" down below.
set(KOKKOS_ARCH "NOT_SET" CACHE STRING 
      "Optimize for specific host architecture. Options are: ${KOKKOS_INTERNAL_ARCH_DOCSTR}")

# Whether to build separate libraries or now
set(KOKKOS_SEPARATE_LIBS OFF CACHE BOOL "OFF = kokkos.  ON = kokkoscore, kokkoscontainers, and kokkosalgorithms.")

# Qthreads options.
set(KOKKOS_QTHREADS_DIR "" CACHE PATH "Location of Qthreads library.")

# HPX options.
set(KOKKOS_HPX_DIR "" CACHE PATH "Location of HPX library.")

# Whether to build separate libraries or now
set(KOKKOS_SEPARATE_TESTS OFF CACHE BOOL "Provide unit test targets with finer granularity.")

set(KOKKOS_HWLOC_DIR "" CACHE PATH "Location of hwloc library. (kokkos tpl)")
set(KOKKOS_MEMKIND_DIR "" CACHE PATH "Location of memkind library. (kokkos tpl)")
set(KOKKOS_CUDA_DIR "" CACHE PATH "Location of CUDA library.  Defaults to where nvcc installed.")

# Make sure KOKKOS_ARCH is set to something
IF ("${KOKKOS_ARCH}" STREQUAL "NOT_SET")
  set(KOKKOS_ARCH "None")
ENDIF()

