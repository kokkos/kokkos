# FIXME_CXX17
IF (KOKKOS_CXX_STANDARD STREQUAL 17)
  IF (KOKKOS_CXX_COMPILER_ID STREQUAL GNU AND KOKKOS_CXX_COMPILER_VERSION VERSION_LESS 7)
    MESSAGE(FATAL_ERROR "You have requested C++17 support for GCC ${KOKKOS_CXX_COMPILER_VERSION}. Although CMake has allowed this and GCC accepts -std=c++1z/c++17, GCC < 7 does not properly support *this capture. Please upgrade the compiler if you do need C++17 support.")
  ENDIF()

  IF (KOKKOS_CXX_COMPILER_ID STREQUAL NVIDIA AND KOKKOS_CXX_COMPILER_VERSION VERSION_LESS 11)
    MESSAGE(FATAL_ERROR "You have requested C++17 support for NVCC ${KOKKOS_CXX_COMPILER_VERSION}. NVCC only supports C++17 from version 11 on. Please upgrade the compiler if you need C++17 support.")
  ENDIF()
  IF (KOKKOS_CXX_COMPILER_ID STREQUAL NVIDIA AND KOKKOS_ENABLE_CUDA_CONSTEXPR)
    MESSAGE(WARNING "You have requested -DKokkos_ENABLE_CUDA_CONSTEXPR=ON with C++17 support for NVCC ${KOKKOS_CXX_COMPILER_VERSION} which is known to trigger compiler bugs. See https://github.com/kokkos/kokkos/issues/3496")
  ENDIF()
ENDIF()

