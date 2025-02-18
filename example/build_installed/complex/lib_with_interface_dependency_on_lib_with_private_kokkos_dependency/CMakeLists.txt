add_library(
  ${PROJECT_NAME}_Lib_with_interface_dependency_on_lib_with_private_kokkos_dependency
        INTERFACE
)
target_include_directories(
  ${PROJECT_NAME}_Lib_with_interface_dependency_on_lib_with_private_kokkos_dependency
  INTERFACE ../../example_source/lib_with_interface_dependency_on_lib_with_private_kokkos_dependency
)
target_link_libraries(
  ${PROJECT_NAME}_Lib_with_interface_dependency_on_lib_with_private_kokkos_dependency
  INTERFACE ${PROJECT_NAME}_Lib_with_private_kokkos_dependency
)