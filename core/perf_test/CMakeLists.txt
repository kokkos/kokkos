# FIXME_OPENMPTARGET - the NVIDIA HPC compiler nvc++ in the OpenMPTarget backend does not pass the perf_tests.
# FIXME_OPENACC - temporarily disabled due to unimplemented features
IF ((KOKKOS_ENABLE_OPENMPTARGET OR KOKKOS_ENABLE_OPENACC) AND KOKKOS_CXX_COMPILER_ID STREQUAL NVHPC)
  RETURN()
ENDIF()

# all PerformanceTest_* executables are part of regular tests
# TODO: finish converting these into benchmarks (in progress)
IF(KOKKOS_ENABLE_TESTS)
  IF(KOKKOS_ENABLE_CUDA OR KOKKOS_ENABLE_HIP OR KOKKOS_ENABLE_SYCL)
    KOKKOS_ADD_EXECUTABLE (
      PerformanceTest_SharedSpace
      SOURCES test_sharedSpace.cpp
    )
  ENDIF()

  KOKKOS_INCLUDE_DIRECTORIES(REQUIRED_DURING_INSTALLATION_TESTING ${CMAKE_CURRENT_SOURCE_DIR})

  IF(NOT Kokkos_ENABLE_OPENMPTARGET)
  # FIXME OPENMPTARGET needs tasking
    KOKKOS_ADD_EXECUTABLE_AND_TEST(
      PerformanceTest_TaskDag
      SOURCES test_taskdag.cpp
      CATEGORIES PERFORMANCE
    )
  ENDIF()
ENDIF()

SET(
  BENCHMARK_SOURCES
  PerfTestGramSchmidt.cpp
  PerfTest_CustomReduction.cpp
  PerfTest_ExecSpacePartitioning.cpp
  PerfTestHexGrad.cpp
  PerfTest_ViewAllocate.cpp
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
  PerfTest_ViewCopy_Raw.cpp
  PerfTest_ViewFill_123.cpp
  PerfTest_ViewFill_45.cpp
  PerfTest_ViewFill_6.cpp
  PerfTest_ViewFill_7.cpp
  PerfTest_ViewFill_8.cpp
  PerfTest_ViewFill_Raw.cpp
  PerfTest_ViewResize_123.cpp
  PerfTest_ViewResize_45.cpp
  PerfTest_ViewResize_6.cpp
  PerfTest_ViewResize_7.cpp
  PerfTest_ViewResize_8.cpp
  PerfTest_ViewResize_Raw.cpp
)

IF(Kokkos_ENABLE_OPENMPTARGET)
# FIXME OPENMPTARGET requires TeamPolicy Reductions and Custom Reduction
  LIST(REMOVE_ITEM BENCHMARK_SOURCES
    PerfTestGramSchmidt.cpp
    PerfTest_CustomReduction.cpp
    PerfTest_ExecSpacePartitioning.cpp
  )
ENDIF()

KOKKOS_ADD_BENCHMARK(
  PerformanceTest_Benchmark
  SOURCES ${BENCHMARK_SOURCES}
)

IF(NOT KOKKOS_ENABLE_CUDA OR KOKKOS_ENABLE_CUDA_LAMBDA)
  KOKKOS_ADD_BENCHMARK(
    Benchmark_Atomic_MinMax
    SOURCES test_atomic_minmax_simple.cpp
  )
ENDIF()

# FIXME_NVHPC
IF(NOT KOKKOS_CXX_COMPILER_ID STREQUAL NVHPC)
  KOKKOS_ADD_BENCHMARK(
    PerformanceTest_Mempool
    SOURCES test_mempool.cpp
  )
ENDIF()

KOKKOS_ADD_BENCHMARK(
  PerformanceTest_Atomic
  SOURCES test_atomic.cpp
)
