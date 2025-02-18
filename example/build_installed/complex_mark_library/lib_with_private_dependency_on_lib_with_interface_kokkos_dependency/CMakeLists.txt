#this include is done to show the difference to the complex example
include(${CMAKE_CURRENT_SOURCE_DIR}/../../complex/lib_with_private_dependency_on_lib_with_interface_kokkos_dependency/CMakeLists.txt)

# as the lib uses Kokkos in its implementation (via the lib it depends on), the source files have to be marked accordingly
kokkos_set_source_and_target_properties(
        SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../example_source/lib_with_private_dependency_on_lib_with_interface_kokkos_dependency/source_with_private_dependency_on_lib_with_interface_kokkos_dependency.cpp
)
# the target needs the correct linker flags
kokkos_set_source_and_target_properties(
        TARGET ${PROJECT_NAME}_Lib_with_private_dependency_on_lib_with_interface_kokkos_dependency)