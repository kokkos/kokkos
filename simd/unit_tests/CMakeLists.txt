kokkos_include_directories(${KOKKOS_SOURCE_DIR}/simd/unit_tests/include)

if(NOT (Kokkos_ENABLE_CUDA AND WIN32) AND NOT (Kokkos_ENABLE_OPENACC AND KOKKOS_CXX_COMPILER_VERSION VERSION_GREATER
                                                                         24.5)
)
  # FIXME_OPENACC: compiling below fails if NVHPC version > 24.5.
  # Error behavior: a device kernel accesses a undefined global symbol.
  # Error message: parse use of undefined value
  kokkos_add_executable_and_test(UnitTest_SIMD SOURCES UnitTestMain.cpp TestSIMD.cpp)
endif()
