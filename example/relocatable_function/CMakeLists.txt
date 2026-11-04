# Copyright Contributors to the Kokkos project
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

kokkos_include_directories(${CMAKE_CURRENT_BINARY_DIR})
kokkos_include_directories(${CMAKE_CURRENT_SOURCE_DIR})

kokkos_add_executable(example_relocatable_function SOURCES main.cpp functor.cpp)

add_test(NAME Kokkos_Example_RelocatableFunction COMMAND Kokkos_example_relocatable_function)
