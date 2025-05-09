kokkos_add_example_directories(query_device)
if(_DEVICE_PARALLEL STREQUAL "NoTypeDefined"
   OR KOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE
   OR KOKKOS_ENABLE_HIP_RELOCATABLE_DEVICE_CODE
   OR KOKKOS_ENABLE_SYCL_RELOCATABLE_DEVICE_CODE
)
  kokkos_add_example_directories(relocatable_function)
endif()
if(NOT (Kokkos_ENABLE_SYCL) OR Kokkos_ENABLE_SYCL_VIRTUAL_FUNCTIONS)
  kokkos_add_example_directories(virtual_functions)
endif()
kokkos_add_example_directories(tutorial)
