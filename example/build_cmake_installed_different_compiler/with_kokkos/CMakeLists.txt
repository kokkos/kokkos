add_executable(example_with_kokkos foo.cpp)
# This is the only thing required to set up compiler/linker flags
target_link_libraries(example_with_kokkos Kokkos::kokkos)
