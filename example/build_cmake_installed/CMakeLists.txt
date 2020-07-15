# Kokkos minimally requires 3.10 right now,
# but your project can set it higher
cmake_minimum_required(VERSION 3.10)

# Declare your project - must have C++ support
project(Example CXX Fortran)

# Look for an installed Kokkos
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.12.0")
  message(STATUS "Setting policy CMP0074 to use <Package>_ROOT variables")
  cmake_policy(SET CMP0074 NEW)
endif()
find_package(Kokkos REQUIRED)

add_executable(example cmake_example.cpp foo.f)
target_link_libraries(example Kokkos::kokkos)

enable_testing()
add_test(NAME KokkosInTree_Verify COMMAND example 10)
