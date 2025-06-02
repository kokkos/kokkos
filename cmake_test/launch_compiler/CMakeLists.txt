# Kokkos minimally requires 3.21 right now,
# but your project can set it higher
cmake_minimum_required(VERSION 3.21)

# Projects can safely mix languages - must have C++ support
# Kokkos flags will only apply to C++ files
project(KokkosCMakeTestingLaunchCompiler CXX Fortran)

# this does not work when compiling with language support (Kokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE)
find_package(Kokkos REQUIRED COMPONENTS launch_compiler)

add_executable(${PROJECT_NAME}_with_kokkos test_with_kokkos.cpp source_plain_fortran.f)
add_executable(${PROJECT_NAME}_no_kokkos test_no_kokkos.cpp source_plain_fortran.f)
if(CMAKE_Fortran_COMPILER_ID STREQUAL LLVMFlang)
  set_target_properties(${PROJECT_NAME}_with_kokkos PROPERTIES LINKER_LANGUAGE Fortran)
  set_target_properties(${PROJECT_NAME}_no_kokkos PROPERTIES LINKER_LANGUAGE Fortran)
  if(CMAKE_Fortran_COMPILER_VERSION VERSION_LESS 19)
    target_link_options(${PROJECT_NAME}_with_kokkos PRIVATE -fno-fortran-main)
    target_link_options(${PROJECT_NAME}_no_kokkos PRIVATE -fno-fortran-main)
  endif()
endif()

# This is the only thing required to set up compiler/linker flags
target_link_libraries(${PROJECT_NAME}_with_kokkos Kokkos::kokkos)

enable_testing()
add_test(NAME ExampleDifferentCompiler_Kokkos_Verify COMMAND ${PROJECT_NAME}_with_kokkos 10)
add_test(NAME ExampleDifferentCompiler_NoKokkos_Verify COMMAND ${PROJECT_NAME}_no_kokkos 10)
