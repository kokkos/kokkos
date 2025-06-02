# Kokkos minimally requires 3.21 right now,
# but your project can set it higher
cmake_minimum_required(VERSION 3.21)

# Projects can safely mix languages - must have C++ support
project(KokkosCMakeTesting CXX)

set(Kokkos_EXCLUDE_BUILD_DIRS
  "${CMAKE_CURRENT_SOURCE_DIR}/complex_mark_library;${CMAKE_CURRENT_SOURCE_DIR}/launch_compiler"
    CACHE
      STRING
      "List of directories Kokkos that are excluded from Kokkos setting build properties on source and target properties via a deferred call"
)

enable_testing()
add_subdirectory(complex)
add_subdirectory(complex_mark_kokkos_files)
add_subdirectory(complex_mark_library)
add_subdirectory(launch_compiler)
