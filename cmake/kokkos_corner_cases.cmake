if(KOKKOS_CXX_COMPILER_ID STREQUAL NVIDIA AND KOKKOS_ENABLE_CUDA_CONSTEXPR AND KOKKOS_CXX_COMPILER_VERSION VERSION_LESS
                                                                               11.2
)
  message(
    WARNING
      "You have requested -DKokkos_ENABLE_CUDA_CONSTEXPR=ON for NVCC ${KOKKOS_CXX_COMPILER_VERSION} which is known to trigger compiler bugs before NVCC version 11.2. See https://github.com/kokkos/kokkos/issues/3496"
  )
endif()
