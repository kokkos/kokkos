#this include is done to show the difference to the complex test
include(
  ${CMAKE_CURRENT_SOURCE_DIR}/../../complex/lib_with_private_dependency_on_lib_with_interface_kokkos_dependency/CMakeLists.txt
)

# the target needs the correct linker flags
kokkos_set_target_properties(
  TARGET ${PROJECT_NAME}_Lib_with_private_dependency_on_lib_with_interface_kokkos_dependency
)
