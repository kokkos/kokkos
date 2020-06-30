# Kokkos minimally requires 3.10 right now,
# but your project can set it higher
cmake_minimum_required(VERSION 3.10)

# Declare your project - must have C++ support
project(Example CXX)

add_subdirectory(${Example_SOURCE_DIR}/../.. ${Example_BINARY_DIR}/kokkos)

add_executable(example cmake_example.cpp)
target_link_libraries(example Kokkos::kokkos)

enable_testing()
add_test(NAME KokkosInTree_Verify COMMAND example 10)
