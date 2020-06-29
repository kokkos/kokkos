cmake_minimum_required(VERSION 3.10)
project(IntreeKokkos CXX) 

IF(CMAKE_VERSION VERSION_GREATER_EQUAL "3.12.0")
  CMAKE_POLICY(SET CMP0074 NEW)
ENDIF()

if (Kokkos_INSTALL_TESTING)
find_package(Kokkos REQUIRED)
else()
add_subdirectory(kokkos)
endif()

add_executable(testInTree test.cc)

include(CMakePrintHelpers)

target_link_libraries(testInTree PRIVATE Kokkos::kokkos)

enable_testing()
add_test(NAME KokkosInTree_Verify COMMAND testInTree)

