KOKKOS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
KOKKOS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

KOKKOS_ADD_EXECUTABLE(
  example_separable_compilation
  SOURCES main.cpp functor.cpp
)

add_test(NAME KokkosSeparableCompilation COMMAND Kokkos_example_separable_compilation)
