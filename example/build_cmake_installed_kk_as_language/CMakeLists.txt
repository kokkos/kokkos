# Kokkos minimally requires 3.16 right now,
# but your project can set it higher
cmake_minimum_required(VERSION 3.19)

# Projects can safely mix languages - must have C++ support
# Kokkos flags will only apply to C++ files
project(Example CXX Fortran CUDA)

# Look for an installed Kokkos
find_package(Kokkos COMPONENTS separable_compilation)

kokkos_set_source_language(cmake_example.cpp)
add_executable(example cmake_example.cpp bar.cpp foo.f)

kokkos_set_target_properties(example)
target_link_libraries(example Kokkos::kokkos)

enable_testing()
add_test(NAME KokkosInTree_Verify COMMAND example 10)
