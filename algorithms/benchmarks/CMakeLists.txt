include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../unit_tests)

KOKKOS_ADD_BENCHMARK(
  Benchmark_Kokkos_ForEach
  SOURCES
    BenchmarkStdAlgorithmsForEach.cpp
)
