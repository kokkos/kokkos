
mdspan_add_openmp_benchmark(stencil_3d_openmp)
if(OpenMP_CXX_FOUND)
  target_include_directories(stencil_3d_openmp PUBLIC
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/benchmarks/stencil>
  )
endif()
