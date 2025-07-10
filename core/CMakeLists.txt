if(NOT Kokkos_INSTALL_TESTING)
  add_subdirectory(src)
  set(KOKKOS_IMPL_COMPILER_PROVIDED_MDSPAN_HEADER ${KOKKOS_IMPL_COMPILER_PROVIDED_MDSPAN_HEADER} PARENT_SCOPE)
endif()

function(KOKKOS_ADD_BENCHMARK_DIRECTORY DIR_NAME)
  if(NOT Kokkos_ENABLE_BENCHMARKS)
    return()
  endif()

  add_subdirectory(${DIR_NAME})
endfunction()

kokkos_add_test_directories(unit_test)
kokkos_add_benchmark_directory(perf_test)
