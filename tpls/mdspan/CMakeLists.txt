cmake_minimum_required(VERSION 3.14)

project(MDSpan
  VERSION 0.6.0
  LANGUAGES CXX
)

include(GNUInstallDirs)

################################################################################

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

################################################################################

option(MDSPAN_ENABLE_TESTS "Enable tests." Off)
option(MDSPAN_ENABLE_EXAMPLES "Build examples." Off)
option(MDSPAN_ENABLE_BENCHMARKS "Enable benchmarks." Off)
option(MDSPAN_ENABLE_COMP_BENCH "Enable compilation benchmarks." Off)
option(MDSPAN_ENABLE_CUDA "Enable Cuda tests/benchmarks/examples if tests/benchmarks/examples are enabled." Off)
option(MDSPAN_ENABLE_SYCL "Enable SYCL tests/benchmarks/examples if tests/benchmarks/examples are enabled." Off)
option(MDSPAN_ENABLE_HIP "Enable HIP tests/benchmarks/examples if tests/benchmarks/examples are enabled." Off)
option(MDSPAN_ENABLE_OPENMP "Enable OpenMP benchmarks if benchmarks are enabled." On)
option(MDSPAN_USE_SYSTEM_GTEST "Use system-installed GoogleTest library for tests." Off)
option(MDSPAN_USE_SYSTEM_BENCHMARK "Use system-installed Google Benchmark library for benchmarks." On)
option(MDSPAN_INSTALL_STDMODE_HEADERS "Whether to install headers to emulate standard library headers and namespaces" Off)
option(MDSPAN_GENERATE_STD_NAMESPACE_TARGETS "Whether to generate and install targets with the std:: namespace instead of the mdspan:: namespace" Off)

# Option to override which C++ standard to use
set(MDSPAN_CXX_STANDARD DETECT CACHE STRING "Override the default CXX_STANDARD to compile with.")
set_property(CACHE MDSPAN_CXX_STANDARD PROPERTY STRINGS DETECT 14 17 20 23)

option(MDSPAN_ENABLE_CONCEPTS "Try to enable concepts support by giving extra flags." On)

################################################################################

# Decide on the standard to use
if(MDSPAN_CXX_STANDARD STREQUAL "17")
  if("cxx_std_17" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    message(STATUS "Using C++17 standard")
    set(CMAKE_CXX_STANDARD 17)
  else()
    message(FATAL_ERROR "Requested MDSPAN_CXX_STANDARD \"17\" not supported by provided C++ compiler")
  endif()
elseif(MDSPAN_CXX_STANDARD STREQUAL "14")
  if("cxx_std_14" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    message(STATUS "Using C++14 standard")
    set(CMAKE_CXX_STANDARD 14)
  else()
    message(FATAL_ERROR "Requested MDSPAN_CXX_STANDARD \"14\" not supported by provided C++ compiler")
  endif()
elseif(MDSPAN_CXX_STANDARD STREQUAL "20")
  if("cxx_std_20" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    message(STATUS "Using C++20 standard")
    set(CMAKE_CXX_STANDARD 20)
  else()
    message(FATAL_ERROR "Requested MDSPAN_CXX_STANDARD \"20\" not supported by provided C++ compiler")
  endif()
elseif(MDSPAN_CXX_STANDARD STREQUAL "23")
  if("cxx_std_23" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    message(STATUS "Using C++23 standard")
    set(CMAKE_CXX_STANDARD 23)
  else()
    message(FATAL_ERROR "Requested MDSPAN_CXX_STANDARD \"23\" not supported by provided C++ compiler")
  endif()
else()
  if("cxx_std_23" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    set(CMAKE_CXX_STANDARD 23)
    message(STATUS "Detected support for C++23 standard")
  elseif("cxx_std_20" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    set(CMAKE_CXX_STANDARD 20)
    message(STATUS "Detected support for C++20 standard")
  elseif("cxx_std_17" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    set(CMAKE_CXX_STANDARD 17)
    message(STATUS "Detected support for C++17 standard")
  elseif("cxx_std_14" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    set(CMAKE_CXX_STANDARD 14)
    message(STATUS "Detected support for C++14 standard")
  else()
    message(FATAL_ERROR "Cannot detect CXX_STANDARD of C++14 or newer.")
  endif()
endif()

################################################################################

if(MDSPAN_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    message(STATUS "Using ${CMAKE_CXX_STANDARD} as CMAKE_CUDA_STANDARD")
    set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    enable_language(CUDA)
  else()
    message(FATAL_ERROR "Requested CUDA support, but no CMAKE_CUDA_COMPILER available")
  endif()
  if(MDSPAN_ENABLE_TESTS)
    set(MDSPAN_TEST_LANGUAGE CUDA)
  endif()
endif()

if(MDSPAN_ENABLE_HIP)
  include(CheckLanguage)
  check_language(HIP)
  if(CMAKE_HIP_COMPILER)
    message(STATUS "Using ${CMAKE_CXX_STANDARD} as CMAKE_HIP_STANDARD")
    set(CMAKE_HIP_STANDARD ${CMAKE_CXX_STANDARD})
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    enable_language(HIP)
  else()
    message(FATAL_ERROR "Requested HIP support, but no CMAKE_HIP_COMPILER available")
  endif()
  if(MDSPAN_ENABLE_TESTS)
    set(MDSPAN_TEST_LANGUAGE HIP)
  endif()
endif()

################################################################################

add_library(mdspan INTERFACE)
add_library(mdspan::mdspan ALIAS mdspan)
if (MDSPAN_GENERATE_STD_NAMESPACE_TARGETS)
  add_library(std::mdspan ALIAS mdspan)
endif()

if(MDSPAN_ENABLE_SYCL)
  target_compile_options(mdspan INTERFACE "-fsycl")
  target_link_options(mdspan INTERFACE "-fsycl")
endif()

target_include_directories(mdspan INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

################################################################################

install(TARGETS mdspan EXPORT mdspanTargets
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT mdspanTargets
    FILE mdspanTargets.cmake
    NAMESPACE mdspan::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mdspan
)

export(TARGETS mdspan
    NAMESPACE mdspan::
    FILE mdspanTargets.cmake
)

if (MDSPAN_GENERATE_STD_NAMESPACE_TARGETS)
  install(TARGETS mdspan EXPORT mdspanStdTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  install(EXPORT mdspanStdTargets
      FILE mdspanStdTargets.cmake
      NAMESPACE std::
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mdspan
  )

  export(TARGETS mdspan
      NAMESPACE std::
      FILE mdspanStdTargets.cmake
  )
endif()

install(DIRECTORY include/mdspan DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
if (MDSPAN_INSTALL_STDMODE_HEADERS)
  install(DIRECTORY include/experimental DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
else()
  install(DIRECTORY include/experimental DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
          REGEX "/mdspan$|/mdarray$" EXCLUDE)
endif()

include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/mdspanConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/mdspanConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mdspan
)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/mdspanConfigVersion.cmake
  COMPATIBILITY SameMajorVersion
  ARCH_INDEPENDENT
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mdspanConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/mdspanConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mdspan
)

################################################################################

if(MDSPAN_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(tests)
  add_subdirectory(compilation_tests)
endif()

if(MDSPAN_ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()

if(MDSPAN_ENABLE_BENCHMARKS)
  if(NOT MDSPAN_CXX_STANDARD STREQUAL "14" AND NOT MDSPAN_CXX_STANDARD STREQUAL "23")
    add_subdirectory(benchmarks)
  else()
    MESSAGE(FATAL_ERROR "Benchmarks are not available in C++14 or C++23 mode. Turn MDSPAN_ENABLE_BENCHMARKS OFF or use C++17 or C++20.")
  endif()
endif()

if(MDSPAN_ENABLE_COMP_BENCH)
  add_subdirectory(comp_bench)
endif()
