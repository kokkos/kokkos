set(KOKKOS_SIMD_TENSOR_CORE_EXAMPLE_BACKEND)
if(Kokkos_ENABLE_EXPERIMENTAL_SIMD_AMX)
  set(KOKKOS_SIMD_TENSOR_CORE_EXAMPLE_BACKEND AMX)
elseif(Kokkos_ENABLE_CUDA)
  set(KOKKOS_SIMD_TENSOR_CORE_EXAMPLE_BACKEND CUDA)
elseif(Kokkos_ENABLE_HIP)
  set(KOKKOS_SIMD_TENSOR_CORE_EXAMPLE_BACKEND HIP)
endif()

if(KOKKOS_SIMD_TENSOR_CORE_EXAMPLE_BACKEND AND NOT WIN32)
  kokkos_add_executable(
    example_simd_tensor_core_shared_memory_matmul
    SOURCES shared_memory_matmul.cpp
  )
  kokkos_add_executable(
    example_simd_tensor_core_tensor_contraction
    SOURCES tensor_contraction.cpp
  )
  kokkos_add_executable(
    example_simd_tensor_core_ozaki2_dgemm
    SOURCES ozaki2_dgemm.cpp
  )

  add_test(
    NAME Kokkos_Example_SIMD_TensorCore_SharedMemoryMatmul
    COMMAND Kokkos_example_simd_tensor_core_shared_memory_matmul
  )
  add_test(
    NAME Kokkos_Example_SIMD_TensorCore_TensorContraction
    COMMAND Kokkos_example_simd_tensor_core_tensor_contraction
  )
  add_test(
    NAME Kokkos_Example_SIMD_TensorCore_Ozaki2DGEMM
    COMMAND Kokkos_example_simd_tensor_core_ozaki2_dgemm 128 128 128
  )
endif()
