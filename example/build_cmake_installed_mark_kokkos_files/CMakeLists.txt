# Kokkos minimally requires 3.22 right now,
# but your project can set it higher
cmake_minimum_required(VERSION 3.22)

# Projects can safely mix languages - must have C++ support
# Kokkos flags will only apply to C++ files
project(Example CXX Fortran)

find_package(Kokkos)

# do not use the compiler launcher globally
kokkos_set_launch_compiler(GLOBAL OFF)
add_subdirectory(no_kokkos)
add_subdirectory(with_kokkos)

enable_testing()
add_test(NAME KokkosLauncher_NoKokkos_Verify COMMAND example_no_kokkos 10)
add_test(NAME KokkosLauncher_WithKokkos_Verify COMMAND example_with_kokkos 10)
