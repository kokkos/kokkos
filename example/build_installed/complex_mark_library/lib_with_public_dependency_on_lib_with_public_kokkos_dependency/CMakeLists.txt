#this include is done to show the difference to the complex example
include(${CMAKE_CURRENT_SOURCE_DIR}/../../complex/lib_with_public_dependency_on_lib_with_public_kokkos_dependency/CMakeLists.txt)

# as the lib uses Kokkos in its implementation, the source files have to be marked accordingly
kokkos_set_source_and_target_properties(
        SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../example_source/lib_with_public_dependency_on_lib_with_public_kokkos_dependency/source_with_public_dependency_on_lib_with_public_kokkos_dependency.cpp
)
# same for the target
kokkos_set_source_and_target_properties(
        TARGET ${PROJECT_NAME}_Lib_with_public_dependency_on_lib_with_public_kokkos_dependency
)