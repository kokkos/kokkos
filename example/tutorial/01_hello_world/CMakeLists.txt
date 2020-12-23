# the next for lines are just, so that we can use this as
# a standalone CMakeLists.txt for testing of the Target export
cmake_minimum_required(VERSION 3.16)
project(Kokkos_HelloWorld CXX)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  find_package(Kokkos)
  add_executable(HelloWorld hello_world.cpp)
  target_link_libraries(HelloWorld Kokkos::kokkos)
  return()
endif()

add_executable(HelloWorld hello_world.cpp)
target_link_libraries(HelloWorld Cabana::cabanacore)
add_test(NAME Core_tutorial_01 COMMAND ${NONMPI_PRECOMMAND} HelloWorld)

KOKKOS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
KOKKOS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

# This is a tutorial, not a test, so we don't ask CTest to run it.
KOKKOS_ADD_EXECUTABLE(
  tutorial_01_hello_world
  SOURCES hello_world.cpp
)

