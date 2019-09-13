IF(KOKKOS_CXX_COMPILER_ID STREQUAL Clang AND KOKKOS_ENABLE_OPENMP)
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
  UNSET(CLANG_HAS_OMP CACHE) #make sure CMake always re-runs this
ENDIF()


