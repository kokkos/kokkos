kokkos_include_directories(${CMAKE_CURRENT_BINARY_DIR})
kokkos_include_directories(REQUIRED_DURING_INSTALLATION_TESTING ${CMAKE_CURRENT_SOURCE_DIR})
kokkos_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)

foreach(Tag Threads;OpenMP;Cuda;HPX;HIP)
  string(TOUPPER ${Tag} DEVICE)
  string(TOLOWER ${Tag} dir)

  if(Kokkos_ENABLE_${DEVICE})
    kokkos_add_executable(ContainersPerformanceTest_${Tag} SOURCES Test${Tag}.cpp)
    add_test(NAME ContainersPerformanceTest_${Tag} COMMAND Kokkos_ContainersPerformanceTest_${Tag})
  endif()
endforeach()
