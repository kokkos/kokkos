
FUNCTION(KOKKOS_DEVICE_OPTION SUFFIX DEFAULT DEV_TYPE DOCSTRING)
  KOKKOS_OPTION(ENABLE_${SUFFIX} ${DEFAULT} BOOL ${DOCSTRING})
  STRING(TOUPPER ${SUFFIX} UC_NAME)
  IF (KOKKOS_ENABLE_${UC_NAME})
    LIST(APPEND KOKKOS_ENABLED_DEVICES    ${SUFFIX})
    #I hate that CMake makes me do this
    SET(KOKKOS_ENABLED_DEVICES    ${KOKKOS_ENABLED_DEVICES}    PARENT_SCOPE)
  ENDIF()
  SET(KOKKOS_ENABLE_${UC_NAME} ${KOKKOS_ENABLE_${UC_NAME}} PARENT_SCOPE)
  IF (KOKKOS_ENABLE_${UC_NAME} AND DEV_TYPE STREQUAL "HOST")
    SET(KOKKOS_HAS_HOST ON PARENT_SCOPE)
  ENDIF()
ENDFUNCTION()

KOKKOS_CFG_DEPENDS(DEVICES NONE)

# Put a check in just in case people are using this option
KOKKOS_DEPRECATED_LIST(DEVICES ENABLE)


KOKKOS_DEVICE_OPTION(PTHREAD       OFF HOST "Whether to build Pthread backend")
IF (KOKKOS_ENABLE_PTHREAD)
  #patch the naming here
  SET(KOKKOS_ENABLE_THREADS ON)
ENDIF()

IF(Trilinos_ENABLE_Kokkos AND Trilinos_ENABLE_OpenMP)
  SET(OMP_DEFAULT ON)
ELSE()
  SET(OMP_DEFAULT OFF)
ENDIF()
KOKKOS_DEVICE_OPTION(OPENMP ${OMP_DEFAULT} HOST "Whether to build OpenMP backend")
IF(KOKKOS_ENABLE_OPENMP)
  COMPILER_SPECIFIC_FLAGS(
    Clang      -fopenmp=libomp
    AppleClang -Xpreprocessor -fopenmp
    PGI        -mp
    NVIDIA     -Xcompiler -fopenmp
    Cray       NO-VALUE-SPECIFIED
    XL         -qsmp=omp
    DEFAULT    -fopenmp
  )
  COMPILER_SPECIFIC_LIBS(
    AppleClang -lomp
  )
ENDIF()

KOKKOS_DEVICE_OPTION(OPENMPTARGET OFF DEVICE "Whether to build the OpenMP target backend")
IF (KOKKOS_ENABLE_OPENMPTARGET)
  COMPILER_SPECIFIC_FLAGS(
    Clang      -fopenmp -fopenmp=libomp
    XL         -qsmp=omp -qoffload -qnoeh
    DEFAULT    -fopenmp
  )
  COMPILER_SPECIFIC_DEFS(
    XL    KOKKOS_IBM_XL_OMP45_WORKAROUND
    Clang KOKKOS_WORKAROUND_OPENMPTARGET_CLANG
  )
# Are there compilers which identify as Clang and need this library?
#  COMPILER_SPECIFIC_LIBS(
#    Clang -lopenmptarget
#  )
ENDIF()

IF(Trilinos_ENABLE_Kokkos AND TPL_ENABLE_CUDA)
  SET(CUDA_DEFAULT ON)
ELSE()
  SET(CUDA_DEFAULT OFF)
ENDIF()
KOKKOS_DEVICE_OPTION(CUDA ${CUDA_DEFAULT} DEVICE "Whether to build CUDA backend")

IF (KOKKOS_ENABLE_CUDA)
  GLOBAL_SET(KOKKOS_DONT_ALLOW_EXTENSIONS "CUDA enabled")
  LIST(APPEND DEVICE_SETUP_LIST Cuda)
ENDIF()

# We want this to default to OFF for cache reasons, but if no
# host space is given, then activate serial
IF (KOKKOS_HAS_TRILINOS)
  #However, Trilinos always wants Serial ON
  SET(SERIAL_DEFAULT ON)
ELSEIF (KOKKOS_HAS_HOST)
  SET(SERIAL_DEFAULT OFF)
ELSE()
  SET(SERIAL_DEFAULT ON)
  IF (NOT DEFINED Kokkos_ENABLE_SERIAL)
    MESSAGE(STATUS "SERIAL backend is being turned on to ensure there is at least one Host space. To change this, you must enable another host execution space and configure with -DKokkos_ENABLE_SERIAL=OFF or change CMakeCache.txt")
  ENDIF()
ENDIF()
KOKKOS_DEVICE_OPTION(SERIAL ${SERIAL_DEFAULT} HOST "Whether to build serial backend")

KOKKOS_DEVICE_OPTION(HPX OFF HOST "Whether to build HPX backend (experimental)")

KOKKOS_DEVICE_OPTION(HIP OFF DEVICE "Whether to build HIP backend")

IF (KOKKOS_ENABLE_HIP)
  LIST(APPEND DEVICE_SETUP_LIST HIP)
ENDIF()
SET(DEVICE_SETUP_LIST ${DEVICE_SETUP_LIST} PARENT_SCOPE)
