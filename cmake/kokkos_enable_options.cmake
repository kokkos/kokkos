########################## NOTES ###############################################
#  List the options for configuring kokkos using CMake method of doing it.
#  These options then get mapped onto KOKKOS_SETTINGS environment variable by
#  kokkos_settings.cmake.  It is separate to allow other packages to override
#  these variables (e.g., TriBITS).

########################## AVAILABLE OPTIONS ###################################
# Use lists for documentation, verification, and programming convenience


FUNCTION(KOKKOS_ENABLE_OPTION SUFFIX DEFAULT DOCSTRING)
  KOKKOS_OPTION(ENABLE_${SUFFIX} ${DEFAULT} BOOL ${DOCSTRING})
  STRING(TOUPPER ${SUFFIX} UC_NAME)
  IF (KOKKOS_ENABLE_${UC_NAME})
    LIST(APPEND KOKKOS_ENABLED_OPTIONS ${UC_NAME})
    #I hate that CMake makes me do this
    SET(KOKKOS_ENABLED_OPTIONS ${KOKKOS_ENABLED_OPTIONS} PARENT_SCOPE)
  ENDIF()
  SET(KOKKOS_ENABLE_${UC_NAME} ${KOKKOS_ENABLE_${UC_NAME}} PARENT_SCOPE)
ENDFUNCTION()

# Certain defaults will depend on knowing the enabled devices
KOKKOS_CFG_DEPENDS(OPTIONS DEVICES)

# Put a check in just in case people are using this option
KOKKOS_DEPRECATED_LIST(OPTIONS ENABLE)

KOKKOS_OPTION(SEPARATE_LIBS  OFF BOOL "whether to build libkokkos or libkokkoscontainers, etc")
KOKKOS_ENABLE_OPTION(CUDA_RELOCATABLE_DEVICE_CODE  OFF "Whether to enable relocatable device code (RDC) for CUDA")
KOKKOS_ENABLE_OPTION(CUDA_UVM             OFF "Whether to enable unified virtual memory (UVM) for CUDA")
KOKKOS_ENABLE_OPTION(CUDA_LDG_INTRINSIC   OFF "Whether to use CUDA LDG intrinsics")
KOKKOS_ENABLE_OPTION(HPX_ASYNC_DISPATCH   OFF "Whether HPX supports asynchronous dispath")
KOKKOS_ENABLE_OPTION(TESTS         OFF  "Whether to build serial  backend")
KOKKOS_ENABLE_OPTION(EXAMPLES      OFF  "Whether to build OpenMP  backend")
STRING(TOUPPER "${CMAKE_BUILD_TYPE}" UPPERCASE_CMAKE_BUILD_TYPE)
IF(UPPERCASE_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
  KOKKOS_ENABLE_OPTION(DEBUG                ON "Whether to activate extra debug features - may increase compile times")
  KOKKOS_ENABLE_OPTION(DEBUG_DUALVIEW_MODIFY_CHECK ON "Debug check on dual views")
ELSE()
  KOKKOS_ENABLE_OPTION(DEBUG                OFF "Whether to activate extra debug features - may increase compile times")
  KOKKOS_ENABLE_OPTION(DEBUG_DUALVIEW_MODIFY_CHECK OFF "Debug check on dual views")
ENDIF()
UNSET(_UPPERCASE_CMAKE_BUILD_TYPE)
KOKKOS_ENABLE_OPTION(LARGE_MEM_TESTS      OFF "Whether to perform extra large memory tests")
KOKKOS_ENABLE_OPTION(DEBUG_BOUNDS_CHECK   OFF "Whether to use bounds checking - will increase runtime")
KOKKOS_ENABLE_OPTION(COMPILER_WARNINGS    OFF "Whether to print all compiler warnings")
KOKKOS_ENABLE_OPTION(PROFILING            ON  "Whether to create bindings for profiling tools")
KOKKOS_ENABLE_OPTION(PROFILING_LOAD_PRINT OFF "Whether to print information about which profiling tools got loaded")
KOKKOS_ENABLE_OPTION(AGGRESSIVE_VECTORIZATION OFF "Whether to aggressively vectorize loops")
KOKKOS_ENABLE_OPTION(DEPRECATED_CODE          OFF "Whether to enable deprecated code")
KOKKOS_ENABLE_OPTION(EXPLICIT_INSTANTIATION   OFF 
  "Whether to explicitly instantiate certain types to lower future compile times")
SET(KOKKOS_ENABLE_ETI ${KOKKOS_ENABLE_EXPLICIT_INSTANTIATION} CACHE INTERNAL "eti")

IF (DEFINED CUDA_VERSION AND CUDA_VERSION VERSION_GREATER "7.0")
  SET(LAMBDA_DEFAULT ON)
ELSE()
  SET(LAMBDA_DEFAULT OFF)
ENDIF()
KOKKOS_ENABLE_OPTION(CUDA_LAMBDA ${LAMBDA_DEFAULT} "Whether to activate experimental lambda features")

FUNCTION(check_device_specific_options)
  CMAKE_PARSE_ARGUMENTS(SOME "" "DEVICE" "OPTIONS" ${ARGN})
  IF(NOT KOKKOS_ENABLE_${SOME_DEVICE})
    FOREACH(OPTION ${SOME_OPTIONS})
      IF(CMAKE_VERSION VERSION_GREATER_EQUAL 3.14)
        IF(NOT DEFINED CACHE{Kokkos_ENABLE_${OPTION}} OR NOT DEFINED CACHE{Kokkos_ENABLE_${SOME_DEVICE}})
          MESSAGE(FATAL_ERROR "Internal logic error: option '${OPTION}' or device '${SOME_DEVICE}' not recognized.")
        ENDIF()
      ENDIF()
      IF(KOKKOS_ENABLE_${OPTION})
        MESSAGE(WARNING "Kokkos_ENABLE_${OPTION} is ON but ${SOME_DEVICE} backend is not enabled. Option will be ignored.")
        UNSET(KOKKOS_ENABLE_${OPTION} PARENT_SCOPE)
      ENDIF()
    ENDFOREACH()
  ENDIF()
ENDFUNCTION()

CHECK_DEVICE_SPECIFIC_OPTIONS(DEVICE CUDA OPTIONS CUDA_UVM CUDA_RELOCATABLE_DEVICE_CODE CUDA_LAMBDA CUDA_LDG_INTRINSIC)
CHECK_DEVICE_SPECIFIC_OPTIONS(DEVICE HPX OPTIONS HPX_ASYNC_DISPATCH)

# Needed due to change from deprecated name to new header define name
IF (KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION)
  SET(KOKKOS_OPT_RANGE_AGGRESSIVE_VECTORIZATION ON)
ENDIF()
