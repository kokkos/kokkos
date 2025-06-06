# Find or download google/benchmark library
find_package(benchmark QUIET 1.5.6)
if(benchmark_FOUND)
  message(STATUS "Using google benchmark found in ${benchmark_DIR}")
else()
  message(STATUS "No installed google benchmark found, fetching from GitHub")
  include(FetchContent)
  set(BENCHMARK_ENABLE_TESTING OFF)

  list(APPEND CMAKE_MESSAGE_INDENT "[benchmark] ")
  FetchContent_Declare(
    googlebenchmark
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
    URL https://github.com/google/benchmark/archive/refs/tags/v1.7.1.tar.gz
    URL_HASH MD5=0459a6c530df9851bee6504c3e37c2e7
  )
  FetchContent_MakeAvailable(googlebenchmark)
  list(POP_BACK CMAKE_MESSAGE_INDENT)

  # Suppress clang-tidy diagnostics on code that we do not have control over
  if(CMAKE_CXX_CLANG_TIDY)
    set_target_properties(benchmark PROPERTIES CXX_CLANG_TIDY "")
  endif()

  # FIXME: Check whether the following target_compile_options are needed.
  # If so, clarify why.
  target_compile_options(benchmark PRIVATE -w)
  target_compile_options(benchmark_main PRIVATE -w)
endif()

kokkos_include_directories(${CMAKE_CURRENT_BINARY_DIR})
kokkos_include_directories(REQUIRED_DURING_INSTALLATION_TESTING ${CMAKE_CURRENT_SOURCE_DIR}/include)
kokkos_include_directories(REQUIRED_DURING_INSTALLATION_TESTING ${CMAKE_CURRENT_SOURCE_DIR}/../../core/perf_test)
kokkos_include_directories(REQUIRED_DURING_INSTALLATION_TESTING ${CMAKE_CURRENT_SOURCE_DIR}/../unit_tests/include)

# We cannot use the kokkos_add_benchmark function defined in core/perf_test/CMakeLists.txt
# because we need our own BenchmarMain.cpp file in order to register the benchmarks at runtime.
foreach(BENCH_TYPE Host Device)
  set(BENCHMARK_NAME Kokkos_Simd${BENCH_TYPE}PerformanceTest)
  set(BENCHMARK_SOURCES BenchmarkMain.cpp ../../core/perf_test/Benchmark_Context.cpp)

  add_executable(${BENCHMARK_NAME} ${BENCHMARK_SOURCES})
  target_link_libraries(${BENCHMARK_NAME} PRIVATE benchmark::benchmark Kokkos::kokkos impl_git_version)
  target_include_directories(${BENCHMARK_NAME} SYSTEM PRIVATE ${benchmark_SOURCE_DIR}/include)
  string(TOUPPER "KOKKOS_IMPL_SIMD_${BENCH_TYPE}_PERFTEST" BENCH_TYPE_MACRO)
  target_compile_definitions(${BENCHMARK_NAME} PRIVATE ${BENCH_TYPE_MACRO})

  # FIXME: This alone will not work. It might need an architecture and standard which need to be defined on target level.
  # It will potentially go away with #7582.
  foreach(SOURCE_FILE ${BENCHMARK_SOURCES})
    set_source_files_properties(${SOURCE_FILE} PROPERTIES LANGUAGE ${KOKKOS_COMPILE_LANGUAGE})
  endforeach()

  string(TIMESTAMP BENCHMARK_TIME "%Y-%m-%d_T%H-%M-%S" UTC)
  set(BENCHMARK_ARGS --benchmark_counters_tabular=true --benchmark_out=${BENCHMARK_NAME}_${BENCHMARK_TIME}.json)

  add_test(NAME ${BENCHMARK_NAME} COMMAND ${BENCHMARK_NAME} ${BENCHMARK_ARGS})
endforeach()
