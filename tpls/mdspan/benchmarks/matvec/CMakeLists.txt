
if(MDSPAN_ENABLE_CUDA)
  add_subdirectory(cuda)
endif()

if(MDSPAN_ENABLE_OPENMP)
  add_subdirectory(openmp)
endif()
