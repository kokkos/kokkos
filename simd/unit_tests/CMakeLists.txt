kokkos_include_directories(${KOKKOS_SOURCE_DIR}/simd/unit_tests/include)

if(NOT (Kokkos_ENABLE_CUDA AND WIN32) AND NOT (Kokkos_ENABLE_OPENACC AND KOKKOS_CXX_COMPILER_VERSION VERSION_LESS_EQUAL
                                                                         26.5)
)
  # FIXME_OPENACC: compiling below is known to fail if 24.5 < NVHPC version <= 26.5.
  # Error behavior: a device kernel accesses a undefined global symbol.
  # Error message: parse use of undefined value
  # FIXME_OPENACC: timeout/segfault with 23.7 on X86 but worked on 24.3
  # disabling wholesale for all versions <= 26.5

  set(SIMD_TEST_GROUPS scalar)
  if(KOKKOS_ARCH_AVX512XEON)
    list(APPEND SIMD_TEST_GROUPS avx512)
  elseif(KOKKOS_ARCH_AVX2)
    list(APPEND SIMD_TEST_GROUPS avx2)
  elseif(KOKKOS_ARCH_ARM_SVE)
    list(APPEND SIMD_TEST_GROUPS sve)
  elseif(KOKKOS_ARCH_ARM_NEON)
    list(APPEND SIMD_TEST_GROUPS neon)
  endif()

  set(SIMD_TEST_HEADERS
      ArithmeticOps
      BitwiseOps
      ComparisonOps
      Condition
      Construction
      Conversions
      GeneratorCtors
      InDeviceConstruction
      LoadStore
      MaskOps
      MathOps
      MemoryPermute
      Reductions
      ShiftOps
  )

  set(SIMD_TEST_SOURCES UnitTestMain.cpp)
  foreach(HEADER ${SIMD_TEST_HEADERS})
    foreach(GROUP ${SIMD_TEST_GROUPS})
      if(GROUP STREQUAL "scalar")
        set(SIMD_TEST_ABI_DEFINE KOKKOS_SIMD_TEST_SCALAR)
      else()
        set(SIMD_TEST_ABI_DEFINE KOKKOS_SIMD_TEST_NATIVE)
      endif()
      set(SIMD_TEST_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/TestSIMD_${HEADER}_${GROUP}.cpp)
      file(
        CONFIGURE
        OUTPUT
        ${SIMD_TEST_SOURCE}
        CONTENT
        "#define ${SIMD_TEST_ABI_DEFINE}\n#define KOKKOS_SIMD_TEST_SUITE simd_${GROUP}\n#include <TestSIMD_${HEADER}.hpp>\n"
      )
      list(APPEND SIMD_TEST_SOURCES ${SIMD_TEST_SOURCE})
    endforeach()
  endforeach()

  kokkos_add_executable_and_test(UnitTest_SIMD SOURCES ${SIMD_TEST_SOURCES})
endif()
