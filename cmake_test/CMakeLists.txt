# Kokkos minimally requires 3.21 right now,
# but your project can set it higher
cmake_minimum_required(VERSION 3.21)

# Projects can safely mix languages - must have C++ support
project(KokkosCMakeTesting CXX)

enable_testing()
add_subdirectory(complex)
add_subdirectory(launch_compiler)
