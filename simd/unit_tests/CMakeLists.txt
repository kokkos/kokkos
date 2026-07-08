kokkos_include_directories(${KOKKOS_SOURCE_DIR}/simd/unit_tests/include)

if(NOT (Kokkos_ENABLE_CUDA AND WIN32))
  kokkos_add_executable_and_test(UnitTest_SIMD SOURCES UnitTestMain.cpp TestSIMD.cpp)
endif()

function(kokkos_simd_add_mma_matmul_tests BACKEND EXE_NAME)
  set(SIZES 64 128 256 512 1024 2048)

  foreach(BLOCK IN LISTS ARGN)
    string(REPLACE "," ";" BLOCK_DIMS "${BLOCK}")
    list(GET BLOCK_DIMS 0 BM)
    list(GET BLOCK_DIMS 1 BN)
    list(GET BLOCK_DIMS 2 BK)

    foreach(SIZE IN LISTS SIZES)
      kokkos_add_test(
        NAME UnitTest_SIMD_MMAMatmul_${BACKEND}_M${SIZE}_BM${BM}_BN${BN}_BK${BK}
        EXE ${EXE_NAME}
        ARGS ${SIZE} ${BM} ${BN} ${BK}
      )
    endforeach()
  endforeach()
endfunction()

function(kokkos_simd_add_mma_tensor_contraction_tests BACKEND)
  set(AXES 0 1 2)
  set(LAYOUTS "0,LayoutLeft" "1,LayoutRight")

  foreach(SIZE IN LISTS ARGN)
    foreach(AXIS IN LISTS AXES)
      foreach(LAYOUT IN LISTS LAYOUTS)
        string(REPLACE "," ";" LAYOUT_FIELDS "${LAYOUT}")
        list(GET LAYOUT_FIELDS 0 LAYOUT_ID)
        list(GET LAYOUT_FIELDS 1 LAYOUT_NAME)

        kokkos_add_test(
          NAME UnitTest_SIMD_MMATensorContraction_${BACKEND}_N${SIZE}_Axis${AXIS}_${LAYOUT_NAME}
          EXE UnitTest_SIMD_MMATensorContraction
          ARGS ${SIZE} ${AXIS} ${LAYOUT_ID}
        )
      endforeach()
    endforeach()
  endforeach()
endfunction()

function(kokkos_simd_add_mma_operation_contract_tests BACKEND EXE_NAME)
  foreach(LAYOUT IN ITEMS left right)
    kokkos_add_test(
      NAME UnitTest_SIMD_MMAOperationContract_${BACKEND}_Valid_${LAYOUT}
      EXE ${EXE_NAME}
      ARGS valid-${LAYOUT}
    )
  endforeach()

  kokkos_add_test(
    NAME UnitTest_SIMD_MMAOperationContract_${BACKEND}_Valid_int8_matmul
    EXE ${EXE_NAME}
    ARGS int8-matmul
  )

  if("${BACKEND}" MATCHES "^AMX")
    kokkos_add_test(
      NAME UnitTest_SIMD_MMAOperationContract_${BACKEND}_Valid_distinct_accumulators
      EXE ${EXE_NAME}
      ARGS amx-distinct-accumulators
    )
  endif()
endfunction()

set(KOKKOS_SIMD_MMA_BACKEND)
if(Kokkos_ENABLE_EXPERIMENTAL_SIMD_AMX)
  set(KOKKOS_SIMD_MMA_BACKEND AMX)
elseif(Kokkos_ENABLE_CUDA)
  set(KOKKOS_SIMD_MMA_BACKEND CUDA)
elseif(Kokkos_ENABLE_HIP)
  set(KOKKOS_SIMD_MMA_BACKEND HIP)
endif()

if(KOKKOS_SIMD_MMA_BACKEND AND NOT WIN32)
  kokkos_add_executable(UnitTest_SIMD_MMAMatmul SOURCES mma/matmul.cpp)
  kokkos_add_executable(
    UnitTest_SIMD_MMATensorContraction
    SOURCES mma/tensor_contraction.cpp
  )
  kokkos_add_executable(
    UnitTest_SIMD_MMAOperationContract
    SOURCES mma/operation_contract.cpp
  )

  if(KOKKOS_SIMD_MMA_BACKEND STREQUAL "AMX")
    kokkos_add_executable(
      UnitTest_SIMD_MMAMatmulAMX32
      SOURCES mma/matmul.cpp
    )
    kokkos_add_executable(
      UnitTest_SIMD_MMAOperationContractAMX32
      SOURCES mma/operation_contract.cpp
    )

    target_compile_definitions(Kokkos_UnitTest_SIMD_MMAMatmulAMX32
      PRIVATE
        KOKKOS_SIMD_TEST_MMA_M=32
        KOKKOS_SIMD_TEST_MMA_N=32
        KOKKOS_SIMD_TEST_MMA_K=32
    )
    target_compile_definitions(Kokkos_UnitTest_SIMD_MMAOperationContractAMX32
      PRIVATE
        KOKKOS_SIMD_TEST_MMA_M=32
        KOKKOS_SIMD_TEST_MMA_N=32
        KOKKOS_SIMD_TEST_MMA_K=32
    )

    kokkos_simd_add_mma_matmul_tests(AMX_Shape16x16x32 UnitTest_SIMD_MMAMatmul "64,64,32" "32,64,64")
    kokkos_simd_add_mma_matmul_tests(AMX_Shape32x32x32 UnitTest_SIMD_MMAMatmulAMX32 "64,64,32" "32,64,64")
    kokkos_simd_add_mma_tensor_contraction_tests(AMX 32)
    kokkos_simd_add_mma_operation_contract_tests(AMX_Shape16x16x32 UnitTest_SIMD_MMAOperationContract)
    kokkos_simd_add_mma_operation_contract_tests(AMX_Shape32x32x32 UnitTest_SIMD_MMAOperationContractAMX32)
  elseif(KOKKOS_SIMD_MMA_BACKEND STREQUAL "CUDA")
    kokkos_simd_add_mma_matmul_tests(CUDA UnitTest_SIMD_MMAMatmul "64,32,32" "32,32,32")
    kokkos_simd_add_mma_tensor_contraction_tests(CUDA 8 16 32)
    kokkos_simd_add_mma_operation_contract_tests(CUDA UnitTest_SIMD_MMAOperationContract)
  elseif(KOKKOS_SIMD_MMA_BACKEND STREQUAL "HIP")
    kokkos_simd_add_mma_matmul_tests(HIP UnitTest_SIMD_MMAMatmul "64,64,32" "32,64,32")
    kokkos_simd_add_mma_tensor_contraction_tests(HIP 16 32)
    kokkos_simd_add_mma_operation_contract_tests(HIP UnitTest_SIMD_MMAOperationContract)
  endif()

  set(KOKKOS_SIMD_MMA_TEST_TARGETS
    Kokkos_UnitTest_SIMD_MMAMatmul
    Kokkos_UnitTest_SIMD_MMATensorContraction
    Kokkos_UnitTest_SIMD_MMAOperationContract
  )
  if(KOKKOS_SIMD_MMA_BACKEND STREQUAL "AMX")
    list(APPEND KOKKOS_SIMD_MMA_TEST_TARGETS
      Kokkos_UnitTest_SIMD_MMAMatmulAMX32
      Kokkos_UnitTest_SIMD_MMAOperationContractAMX32
    )
  endif()

  set_property(GLOBAL APPEND PROPERTY KOKKOS_SIMD_MMA_TARGETS
    ${KOKKOS_SIMD_MMA_TEST_TARGETS}
  )

  add_custom_target(Kokkos_UnitTest_SIMD_MMA_All
    DEPENDS ${KOKKOS_SIMD_MMA_TEST_TARGETS}
  )
endif()
