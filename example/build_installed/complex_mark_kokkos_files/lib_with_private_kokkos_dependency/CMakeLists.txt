#this include is done to show the difference to the complex example
include(${CMAKE_CURRENT_SOURCE_DIR}/../../complex/lib_with_private_kokkos_dependency/CMakeLists.txt)

# as the lib uses Kokkos in its implementation, the source files have to be marked accordingly
kokkos_set_source_and_target_properties(
        SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../example_source/lib_with_private_kokkos_dependency/source_with_public_kokkos_dependency.cpp
)
# FIXME: for language builds Kokkos still sets CXX flags which infect non Kokkos files, thus we will need to mark them here
kokkos_set_source_and_target_properties(
        SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../example_source/lib_with_private_kokkos_dependency/source_without_kokkos_dependency.cpp;${CMAKE_CURRENT_SOURCE_DIR}/../../example_source/lib_with_private_kokkos_dependency/source_merging_prints.cpp
)
# same for the target
kokkos_set_source_and_target_properties(
        TARGET ${PROJECT_NAME}_Lib_with_private_kokkos_dependency
)