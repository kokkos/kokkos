#this include is done to show the difference to the complex example
include(${CMAKE_CURRENT_SOURCE_DIR}/../../complex/lib_with_interface_kokkos_dependency/CMakeLists.txt)

# build all targets with kokkos settings if they link to the library downstream
kokkos_defer_set_dependent_library_properties(PUBLIC LIBRARY ${PROJECT_NAME}_Lib_with_interface_kokkos_dependency)