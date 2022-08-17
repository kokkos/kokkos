IF(KOKKOS_CXX_COMPILER_ID STREQUAL Clang AND KOKKOS_ENABLE_OPENMP AND NOT KOKKOS_CLANG_IS_CRAY AND NOT KOKKOS_COMPILER_CLANG_MSVC)
  # The clang "version" doesn't actually tell you what runtimes and tools
  # were built into Clang. We should therefore make sure that libomp
  # was actually built into Clang. Otherwise the user will get nonsensical
  # errors when they try to build.

  #Try compile is the height of CMake nonsense
  #I can't just give it compiler and link flags
  #I have to hackily pretend that compiler flags are compiler definitions
  #and that linker flags are libraries
  #also - this is easier to use than CMakeCheckCXXSourceCompiles
  TRY_COMPILE(CLANG_HAS_OMP
    ${KOKKOS_TOP_BUILD_DIR}/corner_cases
    ${KOKKOS_SOURCE_DIR}/cmake/compile_tests/clang_omp.cpp
    COMPILE_DEFINITIONS -fopenmp=libomp
    LINK_LIBRARIES -fopenmp=libomp
  )
  IF (NOT CLANG_HAS_OMP)
    UNSET(CLANG_HAS_OMP CACHE) #make sure CMake always re-runs this
    MESSAGE(FATAL_ERROR "Clang failed OpenMP check. You have requested -DKokkos_ENABLE_OPENMP=ON, but the Clang compiler does not appear to have been built with OpenMP support")
  ENDIF()
  UNSET(CLANG_HAS_OMP CACHE) #make sure CMake always re-runs this
ENDIF()

IF(KOKKOS_CXX_COMPILER_ID STREQUAL AppleClang AND KOKKOS_ENABLE_OPENMP)
  # The clang "version" doesn't actually tell you what runtimes and tools
  # were built into Clang. We should therefore make sure that libomp
  # was actually built into Clang. Otherwise the user will get nonsensical
  # errors when they try to build.

  #Try compile is the height of CMake nonsense
  #I can't just give it compiler and link flags
  #I have to hackily pretend that compiler flags are compiler definitions
  #and that linker flags are libraries
  #also - this is easier to use than CMakeCheckCXXSourceCompiles
  TRY_COMPILE(APPLECLANG_HAS_OMP
    ${KOKKOS_TOP_BUILD_DIR}/corner_cases
    ${KOKKOS_SOURCE_DIR}/cmake/compile_tests/clang_omp.cpp
    COMPILE_DEFINITIONS -Xpreprocessor -fopenmp
    LINK_LIBRARIES -lomp
  )
  IF (NOT APPLECLANG_HAS_OMP)
    UNSET(APPLECLANG_HAS_OMP CACHE) #make sure CMake always re-runs this
    MESSAGE(FATAL_ERROR "AppleClang failed OpenMP check. You have requested -DKokkos_ENABLE_OPENMP=ON, but the AppleClang compiler does not appear to have been built with OpenMP support")
  ENDIF()
  UNSET(APPLECLANG_HAS_OMP CACHE) #make sure CMake always re-runs this
ENDIF()

