# Kokkos minimally requires 3.21 right now,
# but your project can set it higher
cmake_minimum_required(VERSION 3.21)

# Projects can safely mix languages - must have C++ support
# Kokkos flags will only apply to C++ files
project(ExampleSimple CXX Fortran)

# Look for an installed Kokkos
find_package(Kokkos REQUIRED)

add_executable(ExampleSimple cmake_example.cpp source_plain_fortran.f)
if(CMAKE_Fortran_COMPILER_ID STREQUAL LLVMFlang)
  set_target_properties(ExampleSimple PROPERTIES LINKER_LANGUAGE Fortran)
endif()

# This is the only thing required to set up compiler/linker flags
target_link_libraries(ExampleSimple Kokkos::kokkos)

enable_testing()
add_test(NAME ExampleSimple_Kokkos_Verify COMMAND ExampleSimple 10)
