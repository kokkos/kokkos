add_executable(example_with_kokkos foo.cpp)
# This is the only thing required to set up compiler/linker flags
target_link_libraries(example_with_kokkos Kokkos::kokkos)
# set the compiler launcher for all targets in this source directory
kokkos_set_launch_compiler(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} ON)
