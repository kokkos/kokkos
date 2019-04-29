########################## NOTES ###############################################
#  List the options for configuring kokkos using CMake method of doing it.
#  These options then get mapped onto KOKKOS_SETTINGS environment variable by
#  kokkos_settings.cmake.  It is separate to allow other packages to override
#  these variables (e.g., TriBITS).

########################## AVAILABLE OPTIONS ###################################
# Use lists for documentation, verification, and programming convenience

FUNCTION(KOKKOS_ENABLE_OPTION CAMEL_SUFFIX DEFAULT DOCSTRING)
  kokkos_option(ENABLE_${CAMEL_SUFFIX} ${DEFAULT} BOOL ${DOCSTRING})
ENDFUNCTION(KOKKOS_ENABLE_OPTION)

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
#------------------------------- GENERAL OPTIONS -------------------------------
#-------------------------------------------------------------------------------
# Whether to build separate libraries or now
SET(KOKKOS_SEPARATE_LIBS OFF CACHE BOOL "OFF = kokkos.  ON = kokkoscore, kokkoscontainers, and kokkosalgorithms.")

# Qthreads options.
SET(KOKKOS_QTHREADS_DIR "" CACHE PATH "Location of Qthreads library.")

# HPX options.
SET(KOKKOS_HPX_DIR "" CACHE PATH "Location of HPX library.")

# Whether to build separate libraries or now
SET(KOKKOS_SEPARATE_TESTS OFF CACHE BOOL "Provide unit test targets with finer granularity.")

SET(KOKKOS_HWLOC_DIR "" CACHE PATH "Location of hwloc library. (kokkos tpl)")
SET(KOKKOS_MEMKIND_DIR "" CACHE PATH "Location of memkind library. (kokkos tpl)")
SET(KOKKOS_CUDA_DIR "" CACHE PATH "Location of CUDA library.  Defaults to where nvcc installed.")


