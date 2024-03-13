cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(KokkosCMakeExample CXX)

find_package(Kokkos REQUIRED)

add_executable(cmake_example cmake_example.cpp)

target_link_libraries(cmake_example PUBLIC Kokkos::kokkos)

#Lines below needed for CI testing
enable_testing()
add_test(NAME KokkosInTree_Verify COMMAND example 10)
