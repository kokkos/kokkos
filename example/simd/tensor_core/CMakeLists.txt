set(KOKKOS_SIMD_TENSOR_CORE_EXAMPLE_BACKEND)
if(Kokkos_ENABLE_CUDA)
  set(KOKKOS_SIMD_TENSOR_CORE_EXAMPLE_BACKEND CUDA)
elseif(Kokkos_ENABLE_HIP)
  set(KOKKOS_SIMD_TENSOR_CORE_EXAMPLE_BACKEND HIP)
endif()

if(KOKKOS_SIMD_TENSOR_CORE_EXAMPLE_BACKEND AND NOT WIN32)
  kokkos_add_executable(
    example_simd_tensor_core_shared_memory_matmul
    SOURCES shared_memory_matmul.cpp
  )

  add_test(
    NAME Kokkos_Example_SIMD_TensorCore_SharedMemoryMatmul
    COMMAND Kokkos_example_simd_tensor_core_shared_memory_matmul
  )
endif()
