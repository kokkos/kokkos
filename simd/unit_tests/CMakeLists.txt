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
endfunction()

if(Kokkos_ENABLE_CUDA AND NOT WIN32)
  kokkos_add_executable(UnitTest_SIMD_TensorCoreMatmul SOURCES tensor_core/matmul.cpp)
  kokkos_add_executable(
    UnitTest_SIMD_TensorCoreOperationContract
    SOURCES tensor_core/operation_contract.cpp
  )

  kokkos_simd_add_tensor_core_matmul_tests(CUDA UnitTest_SIMD_TensorCoreMatmul "64,32,32" "32,32,32")
  kokkos_simd_add_tensor_core_operation_contract_tests(CUDA UnitTest_SIMD_TensorCoreOperationContract)
endif()
