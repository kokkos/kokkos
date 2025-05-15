# Kokkos minimally requires 3.16 right now,
# but your project can set it higher
cmake_minimum_required(VERSION 3.16)

# Project can mix languages - must have C++ support
# Kokkos flags are only applied to C++ files
project(Example CXX)

# We build kokkos as a subdirectory of our project
add_subdirectory(${Example_SOURCE_DIR}/../.. ${Example_BINARY_DIR}/kokkos)

add_executable(example cmake_example.cpp)

# This is the only line required to set up all compiler/linker flags
#target_link_libraries(example kokkoscore_module)
target_link_libraries(example Kokkos::kokkos)

if(NOT TARGET kokkoscore_module)
  message(FATAL_ERROR "BLA")
endif()

include(CMakePrintHelpers)
cmake_print_properties(
  TARGETS kokkoscore_module
  PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
)

# Adds a test for the executable
enable_testing()
add_test(NAME KokkosInTree_Verify COMMAND example 10)
