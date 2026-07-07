kokkos_include_directories(${KOKKOS_SOURCE_DIR}/simd/unit_tests/include)

if(NOT (Kokkos_ENABLE_CUDA AND WIN32))
  kokkos_add_executable_and_test(UnitTest_SIMD SOURCES UnitTestMain.cpp TestSIMD.cpp)
endif()

function(kokkos_simd_add_tensor_core_matmul_tests BACKEND EXE_NAME)
  set(SIZES 64 128 256 512 1024 2048)

  foreach(BLOCK IN LISTS ARGN)
    string(REPLACE "," ";" BLOCK_DIMS "${BLOCK}")
    list(GET BLOCK_DIMS 0 BM)
    list(GET BLOCK_DIMS 1 BN)
    list(GET BLOCK_DIMS 2 BK)

    foreach(SIZE IN LISTS SIZES)
      kokkos_add_test(
        NAME UnitTest_SIMD_TensorCoreMatmul_${BACKEND}_M${SIZE}_BM${BM}_BN${BN}_BK${BK}
        EXE ${EXE_NAME}
        ARGS ${SIZE} ${BM} ${BN} ${BK}
      )
    endforeach()
  endforeach()
endfunction()

function(kokkos_simd_add_tensor_core_tensor_contraction_tests BACKEND)
  set(AXES 0 1 2)
  set(LAYOUTS "0,LayoutLeft" "1,LayoutRight")

  foreach(SIZE IN LISTS ARGN)
    foreach(AXIS IN LISTS AXES)
      foreach(LAYOUT IN LISTS LAYOUTS)
        string(REPLACE "," ";" LAYOUT_FIELDS "${LAYOUT}")
        list(GET LAYOUT_FIELDS 0 LAYOUT_ID)
        list(GET LAYOUT_FIELDS 1 LAYOUT_NAME)

        kokkos_add_test(
          NAME UnitTest_SIMD_TensorCoreTensorContraction_${BACKEND}_N${SIZE}_Axis${AXIS}_${LAYOUT_NAME}
          EXE UnitTest_SIMD_TensorCoreTensorContraction
          ARGS ${SIZE} ${AXIS} ${LAYOUT_ID}
        )
      endforeach()
    endforeach()
  endforeach()
endfunction()

function(kokkos_simd_add_tensor_core_operation_contract_tests BACKEND EXE_NAME)
  foreach(LAYOUT IN ITEMS left right)
    kokkos_add_test(
      NAME UnitTest_SIMD_TensorCoreOperationContract_${BACKEND}_Valid_${LAYOUT}
      EXE ${EXE_NAME}
      ARGS valid-${LAYOUT}
    )
  endforeach()

  kokkos_add_test(
    NAME UnitTest_SIMD_TensorCoreOperationContract_${BACKEND}_Valid_int8_matmul
    EXE ${EXE_NAME}
    ARGS int8-matmul
  )

  if("${BACKEND}" MATCHES "^AMX")
    kokkos_add_test(
      NAME UnitTest_SIMD_TensorCoreOperationContract_${BACKEND}_Valid_distinct_accumulators
      EXE ${EXE_NAME}
      ARGS amx-distinct-accumulators
    )
  endif()
endfunction()

set(KOKKOS_SIMD_TENSOR_CORE_BACKEND)
if(Kokkos_ENABLE_EXPERIMENTAL_SIMD_AMX)
  set(KOKKOS_SIMD_TENSOR_CORE_BACKEND AMX)
elseif(Kokkos_ENABLE_CUDA)
  set(KOKKOS_SIMD_TENSOR_CORE_BACKEND CUDA)
elseif(Kokkos_ENABLE_HIP)
  set(KOKKOS_SIMD_TENSOR_CORE_BACKEND HIP)
endif()

if(KOKKOS_SIMD_TENSOR_CORE_BACKEND AND NOT WIN32)
  kokkos_add_executable(UnitTest_SIMD_TensorCoreMatmul SOURCES tensor_core/matmul.cpp)
  kokkos_add_executable(
    UnitTest_SIMD_TensorCoreTensorContraction
    SOURCES tensor_core/tensor_contraction.cpp
  )
  kokkos_add_executable(
    UnitTest_SIMD_TensorCoreOperationContract
    SOURCES tensor_core/operation_contract.cpp
  )

  if(KOKKOS_SIMD_TENSOR_CORE_BACKEND STREQUAL "AMX")
    kokkos_add_executable(
      UnitTest_SIMD_TensorCoreMatmulAMX32
      SOURCES tensor_core/matmul.cpp
    )
    kokkos_add_executable(
      UnitTest_SIMD_TensorCoreOperationContractAMX32
      SOURCES tensor_core/operation_contract.cpp
    )

    target_compile_definitions(Kokkos_UnitTest_SIMD_TensorCoreMatmulAMX32
      PRIVATE
        KOKKOS_SIMD_TEST_WMMA_M=32
        KOKKOS_SIMD_TEST_WMMA_N=32
        KOKKOS_SIMD_TEST_WMMA_K=32
    )
    target_compile_definitions(Kokkos_UnitTest_SIMD_TensorCoreOperationContractAMX32
      PRIVATE
        KOKKOS_SIMD_TEST_WMMA_M=32
        KOKKOS_SIMD_TEST_WMMA_N=32
        KOKKOS_SIMD_TEST_WMMA_K=32
    )

    kokkos_simd_add_tensor_core_matmul_tests(AMX_Shape16x16x32 UnitTest_SIMD_TensorCoreMatmul "64,64,64" "32,64,64")
    kokkos_simd_add_tensor_core_matmul_tests(AMX_Shape32x32x32 UnitTest_SIMD_TensorCoreMatmulAMX32 "64,64,64")
    kokkos_simd_add_tensor_core_tensor_contraction_tests(AMX 32)
    kokkos_simd_add_tensor_core_operation_contract_tests(AMX_Shape16x16x32 UnitTest_SIMD_TensorCoreOperationContract)
    kokkos_simd_add_tensor_core_operation_contract_tests(AMX_Shape32x32x32 UnitTest_SIMD_TensorCoreOperationContractAMX32)
  elseif(KOKKOS_SIMD_TENSOR_CORE_BACKEND STREQUAL "CUDA")
    kokkos_simd_add_tensor_core_matmul_tests(CUDA UnitTest_SIMD_TensorCoreMatmul "64,32,32" "32,32,32")
    kokkos_simd_add_tensor_core_tensor_contraction_tests(CUDA 8 16 32)
    kokkos_simd_add_tensor_core_operation_contract_tests(CUDA UnitTest_SIMD_TensorCoreOperationContract)
  elseif(KOKKOS_SIMD_TENSOR_CORE_BACKEND STREQUAL "HIP")
    kokkos_simd_add_tensor_core_matmul_tests(HIP UnitTest_SIMD_TensorCoreMatmul "64,64,32" "32,64,32")
    kokkos_simd_add_tensor_core_tensor_contraction_tests(HIP 16 32)
    kokkos_simd_add_tensor_core_operation_contract_tests(HIP UnitTest_SIMD_TensorCoreOperationContract)
  endif()
endif()
