add_library(
  ${PROJECT_NAME}_Lib_without_kokkos_dependency
  ../../test_source/lib_without_kokkos_dependency/source_without_kokkos_dependency.cpp
)

target_include_directories(
  ${PROJECT_NAME}_Lib_without_kokkos_dependency PUBLIC ../../test_source/lib_without_kokkos_dependency
)
