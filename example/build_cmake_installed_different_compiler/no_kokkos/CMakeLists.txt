#exclude this directory from launch compiler
kokkos_set_launch_compiler(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OFF)
add_executable(example_no_kokkos bar.cpp)
