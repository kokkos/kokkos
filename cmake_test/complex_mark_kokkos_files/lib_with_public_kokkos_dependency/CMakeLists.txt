#this include is done to show the difference to the complex test
include(${CMAKE_CURRENT_SOURCE_DIR}/../../complex/lib_with_public_kokkos_dependency/CMakeLists.txt)

#for the target
kokkos_set_target_properties(TARGET ${PROJECT_NAME}_Lib_with_public_kokkos_dependency)
