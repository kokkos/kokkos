set(KOKKOS_SIMD_MMA_EXAMPLE_BACKEND)
if(Kokkos_ENABLE_EXPERIMENTAL_SIMD_AMX)
  set(KOKKOS_SIMD_MMA_EXAMPLE_BACKEND AMX)
elseif(Kokkos_ENABLE_CUDA)
  set(KOKKOS_SIMD_MMA_EXAMPLE_BACKEND CUDA)
elseif(Kokkos_ENABLE_HIP)
  set(KOKKOS_SIMD_MMA_EXAMPLE_BACKEND HIP)
endif()

if(KOKKOS_SIMD_MMA_EXAMPLE_BACKEND AND NOT WIN32)
  kokkos_add_executable(
    example_simd_mma_shared_memory_matmul
    SOURCES shared_memory_matmul.cpp
  )
  kokkos_add_executable(
    example_simd_mma_tensor_contraction
    SOURCES tensor_contraction.cpp
  )
  kokkos_add_executable(
    example_simd_mma_ozaki2_dgemm
    SOURCES ozaki2_dgemm.cpp
  )

  add_test(
    NAME Kokkos_Example_SIMD_MMA_SharedMemoryMatmul
    COMMAND Kokkos_example_simd_mma_shared_memory_matmul
  )
  add_test(
    NAME Kokkos_Example_SIMD_MMA_TensorContraction
    COMMAND Kokkos_example_simd_mma_tensor_contraction
  )
  add_test(
    NAME Kokkos_Example_SIMD_MMA_Ozaki2DGEMM
    COMMAND Kokkos_example_simd_mma_ozaki2_dgemm 128 128 128
  )

  set(KOKKOS_SIMD_MMA_EXAMPLE_TARGETS
    Kokkos_example_simd_mma_shared_memory_matmul
    Kokkos_example_simd_mma_tensor_contraction
    Kokkos_example_simd_mma_ozaki2_dgemm
  )

  set_property(GLOBAL APPEND PROPERTY KOKKOS_SIMD_MMA_TARGETS
    ${KOKKOS_SIMD_MMA_EXAMPLE_TARGETS}
  )

  add_custom_target(Kokkos_Example_SIMD_MMA_All
    DEPENDS ${KOKKOS_SIMD_MMA_EXAMPLE_TARGETS}
  )
endif()
