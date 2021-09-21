# Kokkos minimally requires 3.10 right now,
# but your project can set it higher
cmake_minimum_required(VERSION 3.19)

# Projects can safely mix languages - must have C++ support
# Kokkos flags will only apply to C++ files
project(Example CXX Fortran CUDA)

# You need this for using Kokkos_ROOT variable
message(STATUS "Setting policy CMP0074 to use <Package>_ROOT variables")
cmake_policy(SET CMP0074 NEW)

# Look for an installed Kokkos
find_package(Kokkos COMPONENTS separable_compilation)
set_source_files_properties(cmake_example.cpp PROPERTIES LANGUAGE CUDA)
add_executable(example cmake_example.cpp bar.cpp foo.f)

# This is the only thing required to set up compiler/linker flags
target_link_libraries(example Kokkos::kokkos)

enable_testing()
add_test(NAME KokkosInTree_Verify COMMAND example 10)
