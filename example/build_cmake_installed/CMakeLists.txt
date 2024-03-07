cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(KokkosCMakeExample CXX)

find_package(Kokkos REQUIRED)

add_executable(cmake_example cmake_example.cpp)

target_link_libraries(cmake_example PRIVATE Kokkos::kokkos)
