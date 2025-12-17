
mdspan_add_benchmark(sum_3d_right)
mdspan_add_benchmark(sum_3d_left)
mdspan_add_benchmark(sum_submdspan_right)

if(MDSPAN_ENABLE_CUDA)
  add_subdirectory(cuda)
endif()

if(MDSPAN_ENABLE_OPENMP)
  add_subdirectory(openmp)
endif()
