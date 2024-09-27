KOKKOS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
KOKKOS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

KOKKOS_ADD_EXECUTABLE(
  example_relocatable_function
  SOURCES main.cpp functor.cpp
)

add_test(NAME Kokkos_Example_RelocatableFunction COMMAND Kokkos_example_relocatable_function)
