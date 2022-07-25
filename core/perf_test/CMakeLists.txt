
#INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
#INCLUDE_DIRECTORIES(REQUIRED_DURING_INSTALLATION_TESTING ${CMAKE_CURRENT_SOURCE_DIR})


# warning: PerfTest_CustomReduction.cpp uses
# ../../algorithms/src/Kokkos_Random.hpp
# we'll just allow it to be included, but note
# that in TriBITS KokkosAlgorithms can be disabled...
#INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/../../algorithms/src")


IF(NOT benchmark_FOUND)
  SET(BENCHMARK_SOURCE_DIR ${${PARENT_PACKAGE_NAME}_SOURCE_DIR}/tpls/benchmark)

  file(
    GLOB SOURCE_FILES
    ${BENCHMARK_SOURCE_DIR}/src/*.cc
    ${BENCHMARK_SOURCE_DIR}/src/*.h
    ${BENCHMARK_SOURCE_DIR}/include/benchmark/*.h
  )

  #need here for tribits
  KOKKOS_INCLUDE_DIRECTORIES(${BENCHMARK_SOURCE_DIR}/include)
  KOKKOS_ADD_TEST_LIBRARY(
    kokkos_benchmark
    SOURCES ${SOURCE_FILES}
  )

  TARGET_INCLUDE_DIRECTORIES(kokkos_benchmark PUBLIC ${BENCHMARK_SOURCE_DIR}/include)

  # Suppress clang-tidy diagnostics on code that we do not have control over
  IF(CMAKE_CXX_CLANG_TIDY)
    SET_TARGET_PROPERTIES(kokkos_benchmark PROPERTIES CXX_CLANG_TIDY "")
  ENDIF()

  FIND_PACKAGE(Threads QUIET REQUIRED)
  IF(TARGET Threads::Threads)
    SET_TARGET_PROPERTIES(kokkos_benchmark PROPERTIES
                          INTERFACE_LINK_LIBRARIES Threads::Threads)
  ENDIF()
ENDIF()

# FIXME_OPENMPTARGET - the NVIDIA HPC compiler nvc++ in the OpenMPTarget backend does not pass the perf_tests.
# FIXME_OPENACC - temporarily disabled due to unimplemented features
IF ((KOKKOS_ENABLE_OPENMPTARGET OR KOKKOS_ENABLE_OPENACC) AND KOKKOS_CXX_COMPILER_ID STREQUAL NVHPC)
  RETURN()
ENDIF()


SET(SOURCES
  PerfTestMain.cpp
  PerfTestGramSchmidt.cpp
  PerfTestHexGrad.cpp
  PerfTest_CustomReduction.cpp
  PerfTest_ExecSpacePartitioning.cpp
  PerfTest_ViewCopy_a123.cpp
  PerfTest_ViewCopy_b123.cpp
  PerfTest_ViewCopy_c123.cpp
  PerfTest_ViewCopy_d123.cpp
  PerfTest_ViewCopy_a45.cpp
  PerfTest_ViewCopy_b45.cpp
  PerfTest_ViewCopy_c45.cpp
  PerfTest_ViewCopy_d45.cpp
  PerfTest_ViewCopy_a6.cpp
  PerfTest_ViewCopy_b6.cpp
  PerfTest_ViewCopy_c6.cpp
  PerfTest_ViewCopy_d6.cpp
  PerfTest_ViewCopy_a7.cpp
  PerfTest_ViewCopy_b7.cpp
  PerfTest_ViewCopy_c7.cpp
  PerfTest_ViewCopy_d7.cpp
  PerfTest_ViewCopy_a8.cpp
  PerfTest_ViewCopy_b8.cpp
  PerfTest_ViewCopy_c8.cpp
  PerfTest_ViewCopy_d8.cpp
  PerfTest_ViewAllocate.cpp
  PerfTest_ViewFill_123.cpp
  PerfTest_ViewFill_45.cpp
  PerfTest_ViewFill_6.cpp
  PerfTest_ViewFill_7.cpp
  PerfTest_ViewFill_8.cpp
  PerfTest_ViewResize_123.cpp
  PerfTest_ViewResize_45.cpp
  PerfTest_ViewResize_6.cpp
  PerfTest_ViewResize_7.cpp
  PerfTest_ViewResize_8.cpp
  )

IF(Kokkos_ENABLE_OPENMPTARGET)
# FIXME OPENMPTARGET requires TeamPolicy Reductions and Custom Reduction
  LIST(REMOVE_ITEM SOURCES
    PerfTestGramSchmidt.cpp
    PerfTest_CustomReduction.cpp
    PerfTest_ExecSpacePartitioning.cpp
  )
ENDIF()

# Per #374, we always want to build this test, but we only want to run
# it as a PERFORMANCE test.  That's why we separate building the test
# from running the test.

#leave these as basic includes for now
#I don't need anything transitive
KOKKOS_INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/../../algorithms/src")
KOKKOS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
KOKKOS_INCLUDE_DIRECTORIES(REQUIRED_DURING_INSTALLATION_TESTING ${CMAKE_CURRENT_SOURCE_DIR})

# This test currently times out for MSVC
IF(NOT KOKKOS_CXX_COMPILER_ID STREQUAL "MSVC")
  KOKKOS_ADD_EXECUTABLE_AND_TEST(
    PerfTestExec
    SOURCES ${SOURCES}
    CATEGORIES PERFORMANCE
  )
ENDIF()

KOKKOS_ADD_EXECUTABLE_AND_TEST(
  PerformanceTest_Atomic
  SOURCES test_atomic.cpp
  CATEGORIES PERFORMANCE
)

IF(NOT KOKKOS_ENABLE_CUDA OR KOKKOS_ENABLE_CUDA_LAMBDA)
  KOKKOS_ADD_EXECUTABLE_AND_TEST(
    PerformanceTest_Atomic_MinMax
    SOURCES test_atomic_minmax_simple.cpp
    CATEGORIES PERFORMANCE
  )
ENDIF()

# FIXME_NVHPC
IF(NOT KOKKOS_CXX_COMPILER_ID STREQUAL NVHPC)
KOKKOS_ADD_EXECUTABLE_AND_TEST(
  PerformanceTest_Mempool
  SOURCES test_mempool.cpp
  CATEGORIES PERFORMANCE
)
ENDIF()

IF(NOT Kokkos_ENABLE_OPENMPTARGET)
# FIXME OPENMPTARGET needs tasking
  KOKKOS_ADD_EXECUTABLE_AND_TEST(
    PerformanceTest_TaskDag
    SOURCES test_taskdag.cpp
    CATEGORIES PERFORMANCE
  )
ENDIF()

KOKKOS_ADD_EXECUTABLE_AND_TEST(
  PerformanceTest_SampleBenchmark
  SOURCES BenchmarkMain.cpp Benchmark_Sample.cpp
  CATEGORIES PERFORMANCE
)
