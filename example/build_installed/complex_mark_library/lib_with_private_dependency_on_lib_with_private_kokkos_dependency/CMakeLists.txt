#this include is done to show the difference to the complex example
include(${CMAKE_CURRENT_SOURCE_DIR}/../../complex/lib_with_private_dependency_on_lib_with_private_kokkos_dependency/CMakeLists.txt)

# the target needs the correct linker flags
kokkos_set_source_and_target_properties(
        LINK_ONLY_TARGET ${PROJECT_NAME}_Lib_with_private_dependency_on_lib_with_private_kokkos_dependency)