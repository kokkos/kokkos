
mdspan_add_openmp_benchmark(sum_3d_openmp)
if(OpenMP_CXX_FOUND)
  target_include_directories(sum_3d_openmp PUBLIC
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/benchmarks/sum>
  )
endif()