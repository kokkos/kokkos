IF (NOT CUDAToolkit_ROOT)
  IF (NOT CUDA_ROOT)
    SET(CUDA_ROOT $ENV{CUDA_ROOT})
  ENDIF()
  IF(CUDA_ROOT)
    SET(CUDAToolkit_ROOT ${CUDA_ROOT_ENV})
  ENDIF()
ENDIF()

IF(CMAKE_VERSION VERSION_GREATER_EQUAL "3.17.0")
  find_package(CUDAToolkit)
ELSE()
  include(${CMAKE_CURRENT_LIST_DIR}/CudaToolkit.cmake)
ENDIF()


IF (TARGET CUDA::cudart)
  SET(FOUND_CUDART TRUE)
  KOKKOS_EXPORT_IMPORTED_TPL(CUDA::cudart)
ELSE()
  SET(FOUND_CUDART FALSE)
ENDIF()

IF (TARGET CUDA::cuda_driver)
  SET(FOUND_CUDA_DRIVER TRUE)
  KOKKOS_EXPORT_IMPORTED_TPL(CUDA::cuda_driver)
ELSE()
  SET(FOUND_CUDA_DRIVVER FALSE)
ENDIF()

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(TPLCUDA DEFAULT_MSG FOUND_CUDART FOUND_CUDA_DRIVER)
IF (FOUND_CUDA_DRIVER AND FOUND_CUDART)
  KOKKOS_CREATE_IMPORTED_TPL(CUDA INTERFACE
    LINK_LIBRARIES CUDA::cuda_driver CUDA::cudart
  )
ENDIF()
