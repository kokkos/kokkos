function(KOKKOS_ADD_BENCHMARK_DIRECTORY DIR_NAME)
  if(NOT Kokkos_ENABLE_BENCHMARKS)
    return()
  endif()

  if(KOKKOS_HAS_TRILINOS)
    message(
      STATUS
      "Benchmarks are not supported when building as part of Trilinos"
    )
    return()
  endif()

  add_subdirectory(${DIR_NAME})
endfunction()

function(KOKKOS_ADD_BENCHMARK NAME)
  cmake_parse_arguments(
    BENCHMARK
    ""
    ""
    "SOURCES"
    ${ARGN}
  )
  if(DEFINED BENCHMARK_UNPARSED_ARGUMENTS)
    message(
      WARNING
      "Unexpected arguments when adding a benchmark: "
      ${BENCHMARK_UNPARSED_ARGUMENTS}
    )
  endif()

  set(BENCHMARK_NAME ${PACKAGE_NAME}_${NAME})
  list(APPEND BENCHMARK_SOURCES
    ${KOKKOS_SRC_PATH}/core/perf_test/BenchmarkMain.cpp
    ${KOKKOS_SRC_PATH}/core/perf_test/Benchmark_Context.cpp
  )

  add_executable(
    ${BENCHMARK_NAME}
    ${BENCHMARK_SOURCES}
  )
  target_link_libraries(
    ${BENCHMARK_NAME}
    PRIVATE benchmark::benchmark Kokkos::kokkos impl_git_version
  )
  target_include_directories(
    ${BENCHMARK_NAME}
    SYSTEM PRIVATE ${benchmark_SOURCE_DIR}/include
  )

  foreach(SOURCE_FILE ${BENCHMARK_SOURCES})
    set_source_files_properties(
      ${SOURCE_FILE}
      PROPERTIES LANGUAGE ${KOKKOS_COMPILE_LANGUAGE}
    )
  endforeach()

  string(TIMESTAMP BENCHMARK_TIME "%Y-%m-%d_T%H-%M-%S" UTC)
  set(
    BENCHMARK_ARGS
    --benchmark_counters_tabular=true
    --benchmark_out=${BENCHMARK_NAME}_${BENCHMARK_TIME}.json
  )

  add_test(
    NAME ${BENCHMARK_NAME}
    COMMAND ${BENCHMARK_NAME} ${BENCHMARK_ARGS}
  )
endfunction()

if(NOT Kokkos_ENABLE_BENCHMARKS)
  return()
endif()

# Find or download google/benchmark library
find_package(benchmark QUIET)
if(benchmark_FOUND)
  message(STATUS "Using google benchmark found in ${benchmark_DIR}")
else()
  message(STATUS "No installed google benchmark found, fetching from GitHub")
  include(FetchContent)
  set(BENCHMARK_ENABLE_TESTING OFF)

  list(APPEND CMAKE_MESSAGE_INDENT "[benchmark] ")
  FetchContent_Declare(
    googlebenchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.6.2.tar.gz
    URL_HASH MD5=14d14849e075af116143a161bc3b927b
  )
  FetchContent_MakeAvailable(googlebenchmark)
  list(POP_BACK CMAKE_MESSAGE_INDENT)

  # Suppress clang-tidy diagnostics on code that we do not have control over
  if(CMAKE_CXX_CLANG_TIDY)
    set_target_properties(benchmark PROPERTIES CXX_CLANG_TIDY "")
  endif()

  target_compile_options(benchmark PRIVATE -w)
  target_compile_options(benchmark_main PRIVATE -w)
endif()
