kokkos_include_directories(${KOKKOS_SOURCE_DIR}/simd/unit_tests/include)

if(NOT (Kokkos_ENABLE_CUDA AND WIN32) AND NOT (Kokkos_ENABLE_OPENACC AND KOKKOS_CXX_COMPILER_VERSION VERSION_GREATER 24.5))
  kokkos_add_executable_and_test(UnitTest_SIMD SOURCES UnitTestMain.cpp TestSIMD.cpp)
endif()
