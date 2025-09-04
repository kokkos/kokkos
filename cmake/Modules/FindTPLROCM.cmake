include(FindPackageHandleStandardArgs)

find_library(AMD_HIP_LIBRARY amdhip64 PATHS ENV ROCM_PATH PATH_SUFFIXES lib)
find_library(HSA_RUNTIME_LIBRARY hsa-runtime64 PATHS ENV ROCM_PATH PATH_SUFFIXES lib)

kokkos_create_imported_tpl(
  ROCM
  INTERFACE
  LINK_LIBRARIES
  ${HSA_RUNTIME_LIBRARY}
  ${AMD_HIP_LIBRARY}
  COMPILE_DEFINITIONS
  __HIP_ROCclr__
)
