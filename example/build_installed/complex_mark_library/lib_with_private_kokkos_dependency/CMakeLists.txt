#this include is done to show the difference to the complex example
include(${CMAKE_CURRENT_SOURCE_DIR}/../../complex/lib_with_private_kokkos_dependency/CMakeLists.txt)

# mark target as a kokkos target
kokkos_set_source_and_target_properties(TARGET ${PROJECT_NAME}_Lib_with_private_kokkos_dependency)

# link all targets with kokkos settings if they link to the current library downstream
kokkos_defer_set_dependent_library_properties(LIBRARY ${PROJECT_NAME}_Lib_with_private_kokkos_dependency)