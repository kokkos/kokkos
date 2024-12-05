# Kokkos minimally requires 3.16 right now,
# but your project can set it higher
cmake_minimum_required(VERSION 3.19)

# Projects can safely mix languages - must have C++ support
# Kokkos flags will only apply to C++ files
project(Example CXX Fortran)

# Look for an installed Kokkos
find_package(Kokkos COMPONENTS separable_compilation)
# Enable languages used by Kokkos
kokkos_enable_language()

# Mark source files to be compiled in the language that is enabled in Kokkos (if cmake supports that language)
kokkos_set_source_language(cmake_example.cpp)
add_executable(example cmake_example.cpp bar.cpp foo.f)

# Set target to use architecture and std that is used in Kokkos
kokkos_set_target_properties(example)
target_link_libraries(example Kokkos::kokkos)

enable_testing()
add_test(NAME KokkosInTree_Verify COMMAND example 10)
