# Copyright Contributors to the Kokkos project
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

kokkos_include_directories(${CMAKE_CURRENT_BINARY_DIR})
kokkos_include_directories(${CMAKE_CURRENT_SOURCE_DIR})

kokkos_add_executable(example_virtual_functions SOURCES main.cpp classes.cpp)

add_test(NAME Kokkos_Example_VirtualFunctions COMMAND Kokkos_example_virtual_functions)
